package com.shahryar.virtualtryon.phototryon

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/** Photos from the gallery sorted into views by the way the person faces, whatever order they were picked in. */
class UploadSorterTest {

    @Test
    fun eightPhotosPickedInAnyOrderLandInTheirOwnViews() {
        // Picked back first, then sides, then the rest, with a few degrees of error each.
        val angles = listOf(178f, 92f, -88f, 3f, 47f, 131f, -139f, -41f)
        assertArrayEquals(intArrayOf(4, 2, 6, 0, 1, 3, 5, 7), UploadSorter.assign(angles))
    }

    @Test
    fun fourPhotosGoToFrontSidesAndBack() {
        assertArrayEquals(intArrayOf(4, 0, 6, 2), UploadSorter.assign(listOf(-175f, 8f, -95f, 85f)))
    }

    @Test
    fun twoPhotosAimedAtOneViewShareNoView() {
        val views = UploadSorter.assign(listOf(5f, -10f, 180f))
        assertEquals(0, views[0])           // the nearer of the two takes the front
        assertEquals(7, views[1])           // the other the next nearest free view
        assertEquals(4, views[2])
    }

    @Test
    fun photosWithNoReadingTakeTheViewsLeft() {
        assertArrayEquals(intArrayOf(4, 0, 1), UploadSorter.assign(listOf(170f, null, null)))
    }

    @Test
    fun theFaceDecidesFrontAgainstBack() {
        // The shoulders read as facing the camera, but no face: the model swapped left and right on a back.
        assertEquals(170f, UploadSorter.settle(-10f, faceSeen = false), 0.01f)
        // The shoulders read as a back, but a face is there: turned round to the front half.
        assertEquals(-20f, UploadSorter.settle(160f, faceSeen = true), 0.01f)
        // Readings that already agree are left alone.
        assertEquals(15f, UploadSorter.settle(15f, faceSeen = true), 0.01f)
        assertEquals(-170f, UploadSorter.settle(-170f, faceSeen = false), 0.01f)
        // Near side-on the face can go either way: the shoulders are trusted there.
        assertEquals(95f, UploadSorter.settle(95f, faceSeen = true), 0.01f)
        assertEquals(80f, UploadSorter.settle(80f, faceSeen = false), 0.01f)
    }
}
