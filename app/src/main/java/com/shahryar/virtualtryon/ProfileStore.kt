package com.shahryar.virtualtryon

import android.content.Context

/**
 * The user's profile (M7), kept on the phone between runs: what they enter on the profile screen
 * and the face shape Face Mesh measured. Saving is safe from any thread.
 */
class ProfileStore(context: Context) {

    private companion object {
        const val FILE = "profile"
        const val GENDER = "gender"
        const val HEIGHT_CM = "heightCm"
        const val BODY_TYPE = "bodyType"
        const val COLOURS = "colours"
        const val FACE_SHAPE = "faceShape"
        const val FACE_MEASURED = "faceShapeMeasured"
        const val FACE_LENGTH = "faceLengthToWidth"
        const val FACE_FOREHEAD = "faceForeheadToWidth"
        const val FACE_JAW = "faceJawToWidth"
        const val FACE_JAW_ANGLE = "faceJawAngleDeg"
    }

    private val prefs = context.applicationContext.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    fun load(): Profile = Profile(
        gender = Gender.entries.firstOrNull { it.id == prefs.getString(GENDER, null) } ?: Gender.UNSPECIFIED,
        heightCm = prefs.getInt(HEIGHT_CM, -1).takeIf { it > 0 },
        bodyType = BodyType.entries.firstOrNull { it.id == prefs.getString(BODY_TYPE, null) },
        colours = prefs.getStringSet(COLOURS, emptySet())!!.toSet(),
        faceShape = FaceShape.entries.firstOrNull { it.id == prefs.getString(FACE_SHAPE, null) },
    )

    /** Saves everything but the face measurement; [faceMeasured] says whether the face shape came from it. */
    fun save(profile: Profile, faceMeasured: Boolean = isFaceShapeMeasured()) {
        prefs.edit()
            .putString(GENDER, profile.gender.id)
            .putInt(HEIGHT_CM, profile.heightCm ?: -1)
            .putString(BODY_TYPE, profile.bodyType?.id)
            .putStringSet(COLOURS, profile.colours)
            .putString(FACE_SHAPE, profile.faceShape?.id)
            .putBoolean(FACE_MEASURED, faceMeasured && profile.faceShape != null)
            .apply()
    }

    /** Saves a measured face shape: it becomes the profile's face shape, marked as measured. */
    fun saveFaceReading(reading: FaceShapeReading) {
        val p = reading.proportions
        prefs.edit()
            .putString(FACE_SHAPE, reading.shape.id)
            .putBoolean(FACE_MEASURED, true)
            .putFloat(FACE_LENGTH, p.lengthToWidth)
            .putFloat(FACE_FOREHEAD, p.foreheadToWidth)
            .putFloat(FACE_JAW, p.jawToWidth)
            .putFloat(FACE_JAW_ANGLE, p.jawAngleDeg)
            .apply()
    }

    /** The last face measurement, or null if the face has never been measured. */
    fun faceReading(): FaceShapeReading? {
        if (!prefs.contains(FACE_LENGTH)) return null
        val p = FaceProportions(
            prefs.getFloat(FACE_LENGTH, 0f), prefs.getFloat(FACE_FOREHEAD, 0f),
            prefs.getFloat(FACE_JAW, 0f), prefs.getFloat(FACE_JAW_ANGLE, 0f),
        )
        return FaceShapeReading(FaceShapeMeter.classify(p), p)
    }

    /** Whether the profile's face shape is the measured one, rather than picked by the user. */
    fun isFaceShapeMeasured(): Boolean = prefs.getBoolean(FACE_MEASURED, false)
}
