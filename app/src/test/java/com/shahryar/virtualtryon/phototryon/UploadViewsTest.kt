package com.shahryar.virtualtryon.phototryon

import org.junit.Assert.assertEquals
import org.junit.Test

/** Photos chosen from the gallery: which view each stands for, in the order chosen. */
class UploadViewsTest {

    private fun views(count: Int) = Photos.viewsFor(count).map { it.removePrefix("view_").toInt() }

    @Test
    fun eightPhotosAreTheEightViewsInOrder() = assertEquals((0..7).toList(), views(8))

    @Test
    fun fourPhotosAreFrontRightSideBackLeftSide() = assertEquals(listOf(0, 2, 4, 6), views(4))

    @Test
    fun twoPhotosAreFrontAndBack() = assertEquals(listOf(0, 4), views(2))

    @Test
    fun onePhotoIsTheFront() = assertEquals(listOf(0), views(1))

    @Test
    fun otherCountsAreSpreadRoundWithoutRepeats() {
        for (n in 1..8) {
            val v = views(n)
            assertEquals(0, v.first())
            assertEquals("count $n: $v", n, v.toSet().size)
            assertEquals(v.sorted(), v)
        }
    }
}
