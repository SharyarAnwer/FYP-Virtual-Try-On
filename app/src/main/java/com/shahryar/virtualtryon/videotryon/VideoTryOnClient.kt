package com.shahryar.virtualtryon.videotryon

import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** How far a video job has got. */
class VideoJobStatus(val state: String, val stage: String, val error: String?, val elapsed: Double) {
    val finished get() = state == "done" || state == "error"
}

/** The video engine answered with an error, or could not be reached. */
class VideoEngineException(message: String, val code: Int = 0) : IOException(message)

/**
 * Talks to the Video Try-On engine on Modal (photo_tryon/modal/video_engine.py) with plain web requests.
 * Every call blocks: use it off the main thread.
 */
class VideoTryOnClient(address: String) {

    private val base = address.trim().trimEnd('/')

    fun health(): JSONObject = JSONObject(String(call("GET", "/health")))

    fun putVideo(session: String, video: File) {
        call("PUT", "/video/$session", video.readBytes(), "video/mp4", readTimeoutMs = 180_000)
    }

    fun putGarment(id: String, png: ByteArray) {
        call("PUT", "/garment/$id/front", png, "image/png")
    }

    /** Asks for a top, a bottom or both to be put on a session's video; asking twice gives the same job. */
    fun startJob(session: String, top: String?, topName: String?, bottom: String?, bottomName: String?): String {
        val body = JSONObject().put("session", session)
            .put("top", top ?: JSONObject.NULL).put("top_name", topName ?: JSONObject.NULL)
            .put("bottom", bottom ?: JSONObject.NULL).put("bottom_name", bottomName ?: JSONObject.NULL)
        return JSONObject(String(call("POST", "/job", body.toString().toByteArray(), "application/json"))).getString("job")
    }

    fun status(job: String): VideoJobStatus {
        val j = JSONObject(String(call("GET", "/job/$job")))
        return VideoJobStatus(j.getString("state"), j.optString("stage"),
            if (j.isNull("error")) null else j.getString("error"), j.optDouble("elapsed", 0.0))
    }

    /** Fetches one of a job's files into [to], whole or not at all. */
    fun download(job: String, name: String, to: File) {
        val partial = File(to.parentFile, to.name + ".part")
        partial.writeBytes(call("GET", "/job/$job/file/$name", readTimeoutMs = 180_000))
        if (!partial.renameTo(to)) throw VideoEngineException("could not save ${to.name}")
    }

    private fun call(method: String, path: String, body: ByteArray? = null, type: String = "application/octet-stream",
                     readTimeoutMs: Int = 60_000): ByteArray {
        if (!base.startsWith("http")) throw VideoEngineException("The video engine's address is not set")
        val connection = try {
            URL(base + path).openConnection() as HttpURLConnection
        } catch (e: Exception) {
            throw VideoEngineException("The video engine's address is not a web address")
        }
        try {
            connection.requestMethod = method
            connection.connectTimeout = 20_000
            connection.readTimeout = readTimeoutMs
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
                throw VideoEngineException(message, code)
            }
            return connection.inputStream.use { it.readBytes() }
        } catch (e: VideoEngineException) {
            throw e
        } catch (e: IOException) {
            throw VideoEngineException("The video engine could not be reached: ${e.message}")
        } finally {
            connection.disconnect()
        }
    }
}
