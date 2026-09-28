package com.shahryar.virtualtryon

/**
 * MediaPipe's 33 pose landmarks by name, in the order the model returns them.
 *
 * Garment anchors are keyed by these names so artwork files stay readable. The catalog loader
 * uses this table to reject a garment whose parts name a landmark that does not exist, and the
 * overlay uses it to turn names into indices once per garment, rather than on every frame.
 */
object PoseLandmarks {

    private val NAMES = listOf(
        "nose",
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    )

    private val INDEX: Map<String, Int> = NAMES.withIndex().associate { (i, name) -> name to i }

    /** Index of the named landmark, or -1 if MediaPipe has no landmark by that name. */
    fun indexOf(name: String?): Int = name?.let { INDEX[it] } ?: -1
}
