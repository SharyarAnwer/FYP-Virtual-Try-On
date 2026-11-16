package com.shahryar.virtualtryon.phototryon

import android.content.Context
import com.shahryar.virtualtryon.Garment
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Where the engine is. It runs on Modal at a fixed address, built into the app from local.properties;
 * an address set on the profile screen takes its place.
 */
class EngineSettings(context: Context) {
    private val prefs = context.getSharedPreferences("photo_tryon", Context.MODE_PRIVATE)
    var address: String
        get() = prefs.getString("engine_address", "")?.takeIf { it.isNotBlank() } ?: com.shahryar.virtualtryon.BuildConfig.PHOTO_TRYON_ENGINE
        set(value) = prefs.edit().putString("engine_address", value.trim().trimEnd('/')).apply()
}

/** How far a job has got: [parts] are the pieces ready to fetch, in the order they were made. */
class JobStatus(val state: String, val stage: String, val parts: List<String>, val error: String?) {
    val finished get() = state == "done" || state == "error"
}

/** The engine answered with an error, or could not be reached. */
class EngineException(message: String, val code: Int = 0) : IOException(message)

/**
 * Talks to the Photo Try-On engine - the notebook on Kaggle, reached through its tunnel - with
 * plain web requests (stage 4). Every call blocks: use it off the main thread.
 */
class TryOnClient(address: String) {

    companion object {
        /** A job's pieces - the eight views, dressed - and the files they arrive as. */
        val FILES = Photos.VIEWS.associateWith { "$it.jpg" }
    }

    private val base = address.trim().trimEnd('/')

    /** The engine's own account of itself; throws if it cannot be reached. */
    fun health(): JSONObject = JSONObject(String(call("GET", "/health")))

    fun putPhoto(session: String, view: String, jpeg: ByteArray) {
        call("PUT", "/session/$session/$view", jpeg)
    }

    fun putGarment(id: String, face: String, png: ByteArray) {
        call("PUT", "/garment/$id/$face", png)
    }

    /** Asks for an outfit to be put on a session's photos; asking twice gives the same job. */
    fun startJob(session: String, top: String?, bottom: String?): String {
        val body = JSONObject().put("session", session).put("top", top ?: JSONObject.NULL)
            .put("bottom", bottom ?: JSONObject.NULL)
        return JSONObject(String(call("POST", "/job", body.toString().toByteArray(), "application/json"))).getString("job")
    }

    fun status(job: String): JobStatus {
        val j = JSONObject(String(call("GET", "/job/$job")))
        val parts = j.getJSONArray("parts")
        return JobStatus(j.getString("state"), j.optString("stage"), List(parts.length()) { parts.getString(it) },
            if (j.isNull("error")) null else j.getString("error"))
    }

    /** Fetches one of a job's files into [to], whole or not at all. */
    fun download(job: String, name: String, to: File) {
        val partial = File(to.parentFile, to.name + ".part")
        partial.writeBytes(call("GET", "/job/$job/file/$name"))
        if (!partial.renameTo(to)) throw EngineException("could not save ${to.name}")
    }

    private fun call(method: String, path: String, body: ByteArray? = null, type: String = "application/octet-stream"): ByteArray {
        if (!base.startsWith("http")) throw EngineException("The engine's address is not set")
        val connection = try {
            URL(base + path).openConnection() as HttpURLConnection
        } catch (e: Exception) {
            throw EngineException("The engine's address is not a web address")
        }
        try {
            connection.requestMethod = method
            connection.connectTimeout = 15_000
            connection.readTimeout = 120_000
            if (body != null) {
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", type)
                connection.setFixedLengthStreamingMode(body.size)
                connection.outputStream.use { it.write(body) }
            }
            val code = connection.responseCode
            if (code != 200) {
                val text = connection.errorStream?.use { String(it.readBytes()) }.orEmpty()
                val message = runCatching { JSONObject(text).getString("error") }.getOrDefault("the engine answered $code")
                throw EngineException(message, code)
            }
            return connection.inputStream.use { it.readBytes() }
        } catch (e: EngineException) {
            throw e
        } catch (e: IOException) {
            throw EngineException("The engine could not be reached: ${e.message}")
        } finally {
            connection.disconnect()
        }
    }
}

/**
 * Gets an outfit onto a capture's photos: sends the engine whatever it does not have yet - the
 * four photos, the garments' pictures - and asks for the job. Blocking; use off the main thread.
 */
class TryOnRunner(private val context: Context, private val client: TryOnClient, private val photosDir: File) {

    /** The engine's name for this capture: its folder's. */
    private val session = photosDir.name

    fun request(top: Garment?, bottom: Garment?): String {
        val have = client.health().optJSONObject("garments")
        for (garment in listOfNotNull(top, bottom)) {
            val faces = have?.optJSONArray(garment.id)?.let { a -> List(a.length()) { a.getString(it) } }.orEmpty()
            for ((face, path) in listOf("front" to garment.frontPath, "back" to garment.backPath)) {
                if (path != null && face !in faces) client.putGarment(garment.id, face, context.assets.open(path).use { it.readBytes() })
            }
        }
        return try {
            client.startJob(session, top?.id, bottom?.id)
        } catch (e: EngineException) {
            // A fresh engine has never seen this capture: send the photos, then ask again.
            if (e.code != 400) throw e
            for (view in Photos.VIEWS) Photos.jpeg(Photos.file(photosDir, view))?.let { client.putPhoto(session, view, it) }
            client.startJob(session, top?.id, bottom?.id)
        }
    }
}

/** Where an outfit's pieces are kept on the phone, so they are fetched once and shown without the engine after. */
object Results {
    fun dir(photosDir: File, top: Garment?, bottom: Garment?) =
        File(photosDir, "results/${top?.id ?: "none"}__${bottom?.id ?: "none"}").apply { mkdirs() }

    fun file(resultDir: File, part: String) = File(resultDir, TryOnClient.FILES.getValue(part))

    /** True once every view the capture has a photo for is on the phone, dressed. */
    fun complete(photosDir: File, resultDir: File) = Photos.present(photosDir).let { views -> views.isNotEmpty() && views.all { file(resultDir, it).exists() } }
}
