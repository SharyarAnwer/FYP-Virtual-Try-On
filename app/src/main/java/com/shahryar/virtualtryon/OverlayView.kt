package com.shahryar.virtualtryon

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

/**
 * Transparent drawing layer stacked above the camera preview.
 *
 * Draws the garment (M3, articulated in M3.5) and the pose skeleton (M1). The garment is kept
 * behind the wearer's chin. When no pose is detected it falls back to the standing guide, so the
 * screen is never blank.
 *
 * ## The coordinate problem this class exists to solve
 *
 * MediaPipe returns landmarks normalised 0..1 against the *analysis image* (480x640). This view
 * spans the *preview*, which fills a 1080x2460 screen. PreviewView's FILL_CENTER scales the
 * camera image until it covers the view and crops the overflow — on this device that discards
 * roughly 40% of the image width.
 *
 * So `landmark.x * view.width` is wrong: it stretches a 3:4 image across a 21:9 view.
 * [computeMapping] applies the same scale-and-crop PreviewView does, which is why it must stay
 * in step with the preview's scale type.
 *
 * Paints, matrices and buffers are allocated once. onDraw runs every frame, and allocating there
 * causes GC pauses that surface directly as dropped frames.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private companion object {
        /** Landmarks below this confidence are skipped — usually limbs outside the frame. */
        const val MIN_VISIBILITY = 0.5f

        /** Stride of the packed landmark snapshot: x, y, visibility. */
        const val STRIDE = 3

        // MediaPipe pose landmark indices used to place the torso layer.
        const val LEFT_SHOULDER = 11
        const val RIGHT_SHOULDER = 12
        const val LEFT_HIP = 23
        const val RIGHT_HIP = 24
        const val LEFT_ELBOW = 13
        const val RIGHT_ELBOW = 14

        // Leg landmarks, for trousers (M4).
        const val LEFT_KNEE = 25
        const val RIGHT_KNEE = 26
        const val LEFT_ANKLE = 27
        const val RIGHT_ANKLE = 28

        // Face landmarks used to keep the garment behind the chin.
        const val LEFT_EYE = 2
        const val RIGHT_EYE = 5
        const val LEFT_EAR = 7
        const val RIGHT_EAR = 8
        const val MOUTH_LEFT = 9
        const val MOUTH_RIGHT = 10

        /**
         * How far below the mouth the chin reaches, in eye-to-mouth distances. The pose model has
         * no chin point, so the jaw is estimated as an oval from ear to ear whose lowest point is
         * this far below the mouth. 1.2 reaches the bottom of the tester's beard; a clean-shaven
         * chin sits higher, nearer 0.8, where this hides a little of the collar's back that the
         * neck would actually show. Tuned on one person.
         */
        const val CHIN_DROP = 1.2f

        /**
         * The jaw oval's half-width, as a fraction of half the distance between the ear landmarks.
         * The pose model puts its ear points inside the face outline, about a third of the way in
         * from the real ears: on device an oval only as wide as them left the collar's corners
         * showing either side of the beard. 1.5 brings it out to the actual jaw.
         */
        const val JAW_WIDTH = 1.5f

        /**
         * Limits on how far a garment's body is widened or narrowed to fit the wearer, so a bad
         * measurement cannot throw it to a silly size.
         */
        const val MIN_EASE = -0.6f
        const val MAX_EASE = 1.0f

        /**
         * The hip band, for fitting trousers by the hips (FitMode.CHEST): the measuring levels
         * from just above to just below the hip line, on [BodyWidthMeter.LEVELS]'s scale.
         */
        const val HIP_FROM = 0.9f
        const val HIP_TO = 1.1f

        /**
         * How much wider than the wearer's measured torso a garment's body is drawn, by its
         * catalog fit style. The wearer is measured in whatever they have on, so 1.0 would just
         * cover it; these add the garment's own looseness. A property of the garment's cut, not
         * of any one wearer.
         */
        /**
         * The wearer's upper-arm width, as a fraction of their shoulder-joint width: twice
         * [BodyWidthMeter.ARM_HALF_WIDTH], from the tester's M3.5 shots.
         */
        const val ARM_WIDTH_PER_JOINT = 0.24f

        /**
         * Where a raised arm's armpit sits: this x shoulder-joint width below the arm's line,
         * at the shoulder. Judgement, set once for every garment; see [fitSleeve].
         */
        const val ARMPIT_TUCK = 0.18f

        /** The four-corner fit is complete this many degrees past the photo's own sleeve angle. */
        const val RISE_DEG = 30f

        /**
         * How much wider than the arm a sleeve is drawn, by fit style. "regular" is the pink polo's:
         * its photo shows the sleeve as worn, 0.29 joint widths at the cuff, which is 1.2 arm
         * widths. "slim" and "loose" are judgement, not measured.
         */
        fun sleeveAllowance(style: String): Float = when (style) {
            "slim" -> 1.1f
            "loose" -> 1.4f
            else -> 1.2f
        }

        /**
         * How much thicker than the wearer's *measured* arm a sleeve is drawn, by fit style. The
         * arm is measured in whatever sleeve the wearer has on, so these only add the garment's
         * own looseness. Judgement, set once for all garments.
         */
        fun measuredSleeveAllowance(style: String): Float = when (style) {
            "slim" -> 1.0f
            "loose" -> 1.15f
            else -> 1.05f
        }

        fun fitAllowance(style: String): Float = when (style) {
            "slim" -> 1.03f
            "loose" -> 1.15f
            else -> 1.08f
        }

        /**
         * Length of the perpendicular used as a segment's third control point, in artwork px.
         * Only its direction and scale matter; a larger reach keeps the triangle well conditioned.
         */
        const val PERPENDICULAR_REACH = 64f

        /**
         * Exponential smoothing applied to landmark positions: 1.0 follows the model exactly,
         * lower values lag but steady.
         *
         * Raw landmarks jitter a pixel or two every frame, which is invisible on a thin skeleton
         * line and very visible on a garment the size of a torso — the shirt appears to shiver.
         * The cost is responsiveness: the garment trails fast movement slightly. 0.6 keeps the
         * shiver out without a lag you would notice at arm's length.
         */
        const val SMOOTHING = 0.6f

        /**
         * MediaPipe's 33-point pose topology, as index pairs.
         *
         * Spelled out rather than taken from the library's own constant so the skeleton's shape
         * is readable here and cannot shift under a dependency bump.
         */
        val CONNECTIONS = intArrayOf(
            0, 1, 1, 2, 2, 3, 3, 7, 0, 4, 4, 5, 5, 6, 6, 8, 9, 10,
            11, 13, 13, 15, 15, 17, 15, 19, 15, 21, 17, 19,
            12, 14, 14, 16, 16, 18, 16, 20, 16, 22, 18, 20,
            11, 12, 11, 23, 12, 24, 23, 24,
            23, 25, 25, 27, 27, 29, 27, 31, 29, 31,
            24, 26, 26, 28, 28, 30, 28, 32, 30, 32
        )
    }

    /**
     * Everything needed to draw one garment, resolved once when it is selected so that onDraw
     * does no name lookups and no allocation. Immutable after construction, so swapping the
     * whole object is the only way the drawing thread ever sees a change.
     */
    private class GarmentRender(garment: Garment, layers: GarmentLayers) {
        val fills: List<Bitmap> = layers.fills
        val outlines: List<Bitmap?> = layers.outlines
        val seams: List<Bitmap?> = layers.seams
        val count = garment.parts.size
        val isTorso = BooleanArray(count) { garment.parts[it].kind == PartKind.TORSO }
        val fromLandmark = IntArray(count) { PoseLandmarks.indexOf(garment.parts[it].from) }
        val toLandmark = IntArray(count) { PoseLandmarks.indexOf(garment.parts[it].to) }
        val fromAnchor = Array(count) { garment.anchor(garment.parts[it].from) }
        val toAnchor = Array(count) { garment.anchor(garment.parts[it].to) }
        val parentIndex = IntArray(count) { i ->
            garment.parts.indexOfFirst { it.name == garment.parts[i].parent }
        }
        val matrices = Array(count) { Matrix() }

        val leftShoulder = garment.anchor("left_shoulder")
        val rightShoulder = garment.anchor("right_shoulder")
        val leftHip = garment.anchor("left_hip")
        val rightHip = garment.anchor("right_hip")
    }

    /**
     * Everything needed to draw one mesh garment (M3.6). Bone matrices and vertex buffers are
     * allocated here, once, and reused every frame.
     */
    private class MeshRender(
        val front: Bitmap,
        val sleeves: Bitmap,
        val mesh: GarmentMesh,
        val leftShoulder: Anchor,
        val rightShoulder: Anchor,
        val leftHip: Anchor,
        val rightHip: Anchor,
        val leftElbow: Anchor,
        val rightElbow: Anchor,
        /** See [fitAllowance]. */
        val allowance: Float,
        /** Each sleeve's own frame on the photo, or null to follow the anchors; see [SleeveFrame]. */
        val leftSleeve: SleeveFrame?,
        val rightSleeve: SleeveFrame?,
        /** See [sleeveAllowance]. */
        val sleeveAllowance: Float,
        /** See [measuredSleeveAllowance]. */
        val measuredSleeveAllowance: Float
    ) {
        /** Torso, left sleeve, right sleeve — the order of [GarmentMesh.weights]. */
        val bones = Array(GarmentMesh.BONES) { Matrix() }
        val placed = Array(GarmentMesh.BONES) { FloatArray(mesh.vertexCount * 2) }
        val vertices = FloatArray(mesh.vertexCount * 2)

        /**
         * Widening per grid row, set each frame by [fitToBody]. Starts at 0 - the photo's own
         * proportions - until the first body measurement arrives.
         */
        val easeAmount = FloatArray(mesh.rows + 1)
        val widened = FloatArray(mesh.vertexCount * 2)

        /** The wearer's torso width at each [BodyWidthMeter.LEVELS], in view px; NaN if unknown. */
        val levelWidths = FloatArray(BodyWidthMeter.LEVELS.size)
        val chestScratch = FloatArray(BodyWidthMeter.LEVELS.size)

        /**
         * [fitSleeve] output per sleeve: the four corner targets (x, y pairs, 8 per sleeve), how
         * far the armpit moved (2 per sleeve) and how far the fit applies.
         */
        val cornerTargets = FloatArray(16)
        val armpitMove = FloatArray(4)
        val fitAmount = FloatArray(2)
        val scratch = FloatArray(4)

        companion object {
            fun from(garment: Garment, layers: GarmentLayers): MeshRender? {
                return MeshRender(
                    front = layers.meshFront ?: return null,
                    sleeves = layers.meshSleeves ?: return null,
                    mesh = layers.mesh ?: return null,
                    leftShoulder = garment.anchor("left_shoulder") ?: return null,
                    rightShoulder = garment.anchor("right_shoulder") ?: return null,
                    leftHip = garment.anchor("left_hip") ?: return null,
                    rightHip = garment.anchor("right_hip") ?: return null,
                    leftElbow = garment.anchor("left_elbow") ?: return null,
                    rightElbow = garment.anchor("right_elbow") ?: return null,
                    allowance = fitAllowance(garment.fitStyle),
                    leftSleeve = SleeveFrame.of(garment.armholes["left"], garment.anchor("left_shoulder"),
                        garment.anchor("left_elbow")),
                    rightSleeve = SleeveFrame.of(garment.armholes["right"], garment.anchor("right_shoulder"),
                        garment.anchor("right_elbow")),
                    sleeveAllowance = sleeveAllowance(garment.fitStyle),
                    measuredSleeveAllowance = measuredSleeveAllowance(garment.fitStyle)
                )
            }
        }
    }

    /**
     * Everything needed to draw one pair of trousers (M4). Bone matrices and vertex buffers are
     * allocated here, once, and reused every frame.
     */
    private class TrouserRender(
        val front: Bitmap,
        val mesh: TrouserMesh,
        val leftHip: Anchor,
        val rightHip: Anchor,
        val leftKnee: Anchor,
        val rightKnee: Anchor,
        val leftAnkle: Anchor,
        val rightAnkle: Anchor,
        /** See [fitAllowance]. */
        val allowance: Float
    ) {
        /** Pelvis, left leg, right leg — the order of [TrouserMesh.weights]. */
        val bones = Array(TrouserMesh.BONES) { Matrix() }
        val placed = Array(TrouserMesh.BONES) { FloatArray(mesh.vertexCount * 2) }
        val vertices = FloatArray(mesh.vertexCount * 2)

        /**
         * Widening per leg per grid row, outward and inward of each row's pivot, set each frame
         * by [fitTrousers]; see [TrouserMesh.widen]. 0 - the photo's own width - until the first
         * body measurement arrives.
         */
        val outer = Array(2) { FloatArray(mesh.rows + 1) }
        val inner = Array(2) { FloatArray(mesh.rows + 1) }
        val widened = FloatArray(mesh.vertexCount * 2)

        /** The wearer's half-width per side at each [BodyWidthMeter.LEVELS], in view px; NaN if unknown. */
        val levelHalves = Array(2) { FloatArray(BodyWidthMeter.LEVELS.size) }

        /** The wearer's legs at each [BodyWidthMeter.LEG_SAMPLES], outer and inner, in view px. */
        val legOuterView = Array(2) { FloatArray(BodyWidthMeter.LEG_SAMPLES.size) }
        val legInnerView = Array(2) { FloatArray(BodyWidthMeter.LEG_SAMPLES.size) }
        val hipScratch = FloatArray(BodyWidthMeter.LEVELS.size)

        /** Per side, the widening that fits the hips; see [fitTrousers]. */
        val hipEase = FloatArray(2)

        /** The hip anchors' spacing, in artwork px. */
        val hipSpan = hypot(rightHip.x - leftHip.x, rightHip.y - leftHip.y)

        private val leftLeg = hypot(leftAnkle.x - leftHip.x, leftAnkle.y - leftHip.y)
        private val rightLeg = hypot(rightAnkle.x - rightHip.x, rightAnkle.y - rightHip.y)

        /** The two legs' mean length, hip to ankle, in artwork px. */
        val legLength = (leftLeg + rightLeg) / 2f

        /** Per leg, hip to ankle over hip to knee: a wearer's thigh times this is their leg. */
        val legPerThigh = floatArrayOf(
            leftLeg / hypot(leftKnee.x - leftHip.x, leftKnee.y - leftHip.y),
            rightLeg / hypot(rightKnee.x - rightHip.x, rightKnee.y - rightHip.y)
        )

        /** Per leg, the unit vector across it on the photo, square to hip anchor -> ankle anchor. */
        val acrossX = floatArrayOf(-(leftAnkle.y - leftHip.y) / leftLeg, -(rightAnkle.y - rightHip.y) / rightLeg)
        val acrossY = floatArrayOf((leftAnkle.x - leftHip.x) / leftLeg, (rightAnkle.x - rightHip.x) / rightLeg)

        companion object {
            fun from(garment: Garment, layers: GarmentLayers): TrouserRender? {
                return TrouserRender(
                    front = layers.meshFront ?: return null,
                    mesh = layers.trousers ?: return null,
                    leftHip = garment.anchor("left_hip") ?: return null,
                    rightHip = garment.anchor("right_hip") ?: return null,
                    leftKnee = garment.anchor("left_knee") ?: return null,
                    rightKnee = garment.anchor("right_knee") ?: return null,
                    leftAnkle = garment.anchor("left_ankle") ?: return null,
                    rightAnkle = garment.anchor("right_ankle") ?: return null,
                    allowance = fitAllowance(garment.fitStyle)
                )
            }
        }
    }

    /**
     * One sleeve's own axes on the garment photo: its centre line, from the middle of the armhole
     * to the middle of the cuff, and the direction across it. [origin] is the shoulder joint moved
     * onto that line; [length] runs from there to the elbow anchor along it; [width] is the cuff's
     * width across it.
     */
    private class SleeveFrame(
        val originX: Float, val originY: Float,
        val ux: Float, val uy: Float,
        val length: Float, val width: Float,
        /** From [originX], [originY] along the centre line to the cuff's middle. */
        val cuffAlong: Float
    ) {
        companion object {
            fun of(hole: Armhole?, joint: Anchor?, elbow: Anchor?): SleeveFrame? {
                val outer = hole?.cuffOuter ?: return null
                val inner = hole.cuffInner ?: return null
                if (joint == null || elbow == null) return null
                val rootX = (hole.shoulder.x + hole.armpit.x) / 2f
                val rootY = (hole.shoulder.y + hole.armpit.y) / 2f
                var ux = (outer.x + inner.x) / 2f - rootX
                var uy = (outer.y + inner.y) / 2f - rootY
                val n = hypot(ux, uy)
                if (n < 1f) return null
                ux /= n
                uy /= n
                val along = (joint.x - rootX) * ux + (joint.y - rootY) * uy
                val ox = rootX + along * ux
                val oy = rootY + along * uy
                val length = (elbow.x - ox) * ux + (elbow.y - oy) * uy
                val width = abs((outer.x - inner.x) * -uy + (outer.y - inner.y) * ux)
                if (length < 1f || width < 1f) return null
                val cuffAlong = ((outer.x + inner.x) / 2f - ox) * ux + ((outer.y + inner.y) / 2f - oy) * uy
                return SleeveFrame(ox, oy, ux, uy, length, width, cuffAlong)
            }
        }
    }

    private val density = resources.displayMetrics.density

    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#5FD3DB")
    }

    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
    }

    private val bracketPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f * density
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#5FD3DB")
    }

    private val captionPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#CCFFFFFF")
        textSize = 12f * density
        textAlign = Paint.Align.CENTER
        letterSpacing = 0.12f
    }

    // Unlike the 90-degree frame rotation, this scales the artwork by an arbitrary factor, so
    // filtering genuinely changes the result rather than just costing time.
    private val garmentPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        isFilterBitmap = true
    }

    /** Body-width lines: green where measured, amber where filled in from the other side. */
    private val measuredPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f * density
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#3DDC84")
    }
    private val filledPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2.5f * density
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#FFB300")
    }

    private val guide = RectF()
    private val segments = FloatArray(CONNECTIONS.size * 2)
    private val jointRadius = 3.5f * density
    private val mapping = FloatArray(3)

    private val torsoMatrix = Matrix()
    private val torsoInverse = Matrix()
    private val point = FloatArray(2)
    private val pelvisMatrix = Matrix()
    private val pelvisInverse = Matrix()
    private val matrixValues = FloatArray(9)
    private val vector = FloatArray(2)
    private val src = FloatArray(6)
    private val dst = FloatArray(6)

    /** The region the chin and jaw cover, in view coordinates; see [computeFace]. */
    private val facePath = Path()
    private val faceOval = RectF()
    private val faceMatrix = Matrix()

    /** Immutable snapshot written by the analysis thread, read by the UI thread. */
    @Volatile private var pose: FloatArray? = null
    @Volatile private var imageWidth = 0
    @Volatile private var imageHeight = 0

    /** Previous smoothed frame. Analysis thread only. */
    private var previous: FloatArray? = null


    /**
     * Whether to draw the pose skeleton. Off by default: drawn over a garment, its lines and
     * joints are the clearest possible signal that the shirt is an overlay rather than something
     * worn. It stays available for checking alignment, which is what it is for.
     */
    @Volatile var showSkeleton: Boolean = false
        set(value) {
            field = value
            postInvalidate()
        }

    /** The two garments that can be worn together (M4): a shirt and a pair of trousers. */
    enum class Slot { TOP, BOTTOM }

    /** One slot's garment, ready to draw: layered (M3.5), a mesh shirt (M3.6) or trousers (M4). */
    private class Worn(val layered: GarmentRender?, val meshed: MeshRender?, val trousers: TrouserRender?)

    @Volatile private var top: Worn? = null
    @Volatile private var bottom: Worn? = null

    /** Latest body-width snapshot from [BodyWidthMeter], or null. Drawn with the skeleton. */
    @Volatile private var bodyWidths: FloatArray? = null

    /**
     * How a real garment's body is fitted to the wearer's measured width (M3.6). All three are
     * kept so they can be compared on device; the Top fit and Bottom fit buttons switch between
     * them, each for its own garment.
     */
    enum class FitMode {
        /**
         * One width for the whole body: the wearer's width across the chest, just below the
         * armpits, applied to every row. The garment keeps its own shape below that - straight
         * sides stay straight - rather than following the wearer's clothes. Fitted per row, a
         * loose kameez that flares from the armpits made the polo flare and bulge with it.
         *
         * Trousers are fitted the same way by the hips: each side to the wearer's at the hip
         * line, one widening for every row, the legs keeping the garment's own shape.
         */
        CHEST,

        /**
         * Every row fitted to the wearer's width at that height, clothes included. For trousers,
         * the seat row by row, and each leg to its measured thickness down its length.
         */
        ROWS,

        /** No fitting: the garment at its photo's own proportions, scaled to the shoulders. */
        OFF
    }

    /** How the top is fitted; see [FitMode]. */
    @Volatile var topFitMode: FitMode = FitMode.CHEST
        set(value) {
            field = value
            postInvalidate()
        }

    /**
     * How the bottom is fitted; see [FitMode]. Row by row by default, unlike the top: on device,
     * over the tester's loose trousers, fitting the khaki chinos by the hips kept their taper, and
     * his own trousers showed down the outer thighs and between the legs; fitted row by row they
     * were covered in all three poses. One mode for both garments could not suit both.
     */
    @Volatile var bottomFitMode: FitMode = FitMode.ROWS
        set(value) {
            field = value
            postInvalidate()
        }

    /**
     * Mean widening applied over the torso's middle rows in the last frame, for the perf log;
     * NaN until a garment has been fitted.
     */
    @Volatile var fittedEase: Float = Float.NaN
        private set

    /** The same for trousers: mean widening over their rows in the last frame; NaN before any. */
    @Volatile var trouserEase: Float = Float.NaN
        private set

    /** Publishes a body-width snapshot; the next [setPose] triggers the redraw. */
    fun setBodyWidths(snapshot: FloatArray?) {
        bodyWidths = snapshot
    }

    /** Latest arm-thickness snapshot from [BodyWidthMeter.lastArms], or null. */
    @Volatile private var armWidths: FloatArray? = null

    /** Publishes an arm-thickness snapshot; the next [setPose] triggers the redraw. */
    fun setArmWidths(snapshot: FloatArray?) {
        armWidths = snapshot
    }

    /** Latest leg-thickness snapshot from [BodyWidthMeter.lastLegs], or null. */
    @Volatile private var legWidths: FloatArray? = null

    /** Publishes a leg-thickness snapshot; the next [setPose] triggers the redraw. */
    fun setLegWidths(snapshot: FloatArray?) {
        legWidths = snapshot
    }

    /**
     * Sets the garment worn in [slot], or clears it when either argument is null. [layers] must be
     * in the same order as [Garment.parts].
     */
    fun setGarment(slot: Slot, layers: GarmentLayers?, garment: Garment?) {
        var layered: GarmentRender? = null
        var meshed: MeshRender? = null
        var trousers: TrouserRender? = null
        if (layers != null && garment != null) {
            if (layers.trousers != null) {
                trousers = TrouserRender.from(garment, layers)
            } else if (garment.render == RenderKind.MESH) {
                meshed = MeshRender.from(garment, layers)
            } else if (layers.fills.size == garment.parts.size &&
                layers.outlines.size == garment.parts.size && layers.seams.size == garment.parts.size) {
                layered = GarmentRender(garment, layers)
            }
        }
        val worn = if (layered == null && meshed == null && trousers == null) null
            else Worn(layered, meshed, trousers)
        when (slot) {
            Slot.TOP -> top = worn
            Slot.BOTTOM -> bottom = worn
        }
        postInvalidate()
    }

    /**
     * Publishes a new detection. Safe to call from the analysis thread.
     *
     * [imageWidth] and [imageHeight] are the dimensions of the bitmap fed to the model — already
     * upright and mirrored — not the raw sensor frame.
     */
    fun setPose(landmarks: List<NormalizedLandmark>?, imageWidth: Int, imageHeight: Int) {
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight

        val next = landmarks?.takeIf { it.isNotEmpty() }?.let { list ->
            FloatArray(list.size * STRIDE).also { out ->
                list.forEachIndexed { i, lm ->
                    out[i * STRIDE] = lm.x()
                    out[i * STRIDE + 1] = lm.y()
                    out[i * STRIDE + 2] = lm.visibility().orElse(1f)
                }
            }
        }

        if (next == null) {
            previous = null
            pose = null
        } else {
            val prev = previous
            if (prev != null && prev.size == next.size) {
                for (i in next.indices) {
                    next[i] = prev[i] + SMOOTHING * (next[i] - prev[i])
                }
            }
            previous = next
            pose = next
        }
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val snapshot = pose
        if (snapshot == null || imageWidth <= 0 || imageHeight <= 0) {
            drawGuide(canvas)
            return
        }

        computeMapping()
        val faceSeen = computeFace(snapshot)
        val checkpoint = canvas.save()
        if (faceSeen) canvas.clipOutPath(facePath)
        // The bottom first, so a shirt's hem covers the trousers' waistband, as when it is worn out.
        val wearingBottom = bottom?.let { drawGarment(canvas, snapshot, it) } ?: false
        val wearingTop = top?.let { drawGarment(canvas, snapshot, it) } ?: false
        val wearingGarment = wearingBottom || wearingTop
        canvas.restoreToCount(checkpoint)
        if (showSkeleton) {
            drawSkeleton(canvas, snapshot, dimmed = wearingGarment)
            if (faceSeen) canvas.drawPath(facePath, bonePaint)
            bodyWidths?.let { drawBodyWidths(canvas, it) }
            armWidths?.let { drawArmWidths(canvas, it) }
            legWidths?.let { drawLegWidths(canvas, it) }
        }
    }

    /**
     * Sets [facePath] to the region the wearer's chin and jaw cover. Returns false when the face
     * is not seen clearly enough to place it, and then nothing is hidden.
     *
     * The garment is drawn over the camera image, so it covers everything it overlaps. A real
     * shirt's collar stands up around the neck, and drawn on top it covered the tester's jaw and
     * beard up to his mouth. On a person the chin is in front of the collar, so the garment is
     * clipped out of this region.
     *
     * The region is an oval centred between the ears, as wide as they are apart, reaching
     * [CHIN_DROP] below the mouth. Its axis runs perpendicular to the line between the ears,
     * which tilts with the head and is a longer, steadier baseline than eyes to mouth. The eyes
     * and mouth only say which way is down and how long the face is.
     */
    private fun computeFace(p: FloatArray): Boolean {
        if (!visible(p, LEFT_EAR) || !visible(p, RIGHT_EAR) || !visible(p, LEFT_EYE) ||
            !visible(p, RIGHT_EYE) || !visible(p, MOUTH_LEFT) || !visible(p, MOUTH_RIGHT)) return false

        val earX = (viewX(p, LEFT_EAR) + viewX(p, RIGHT_EAR)) / 2f
        val earY = (viewY(p, LEFT_EAR) + viewY(p, RIGHT_EAR)) / 2f
        val acrossX = viewX(p, RIGHT_EAR) - viewX(p, LEFT_EAR)
        val acrossY = viewY(p, RIGHT_EAR) - viewY(p, LEFT_EAR)
        val earSpan = hypot(acrossX, acrossY)
        if (earSpan < 1f) return false

        val eyeX = (viewX(p, LEFT_EYE) + viewX(p, RIGHT_EYE)) / 2f
        val eyeY = (viewY(p, LEFT_EYE) + viewY(p, RIGHT_EYE)) / 2f
        val mouthX = (viewX(p, MOUTH_LEFT) + viewX(p, MOUTH_RIGHT)) / 2f
        val mouthY = (viewY(p, MOUTH_LEFT) + viewY(p, MOUTH_RIGHT)) / 2f

        // Down the face: perpendicular to the ear line, on the mouth's side of the eyes.
        var downX = -acrossY / earSpan
        var downY = acrossX / earSpan
        val eyeToMouth = (mouthX - eyeX) * downX + (mouthY - eyeY) * downY
        if (eyeToMouth < 0f) {
            downX = -downX
            downY = -downY
        }
        val faceLength = abs(eyeToMouth)
        if (faceLength < 1f) return false

        val depth = (mouthX - earX) * downX + (mouthY - earY) * downY + CHIN_DROP * faceLength
        if (depth <= 0f) return false
        val halfWidth = earSpan / 2f * JAW_WIDTH

        faceOval.set(-halfWidth, -depth, halfWidth, depth)
        facePath.rewind()
        facePath.addOval(faceOval, Path.Direction.CW)
        faceMatrix.setRotate(Math.toDegrees(atan2(downY, downX).toDouble()).toFloat() - 90f)
        faceMatrix.postTranslate(earX, earY)
        facePath.transform(faceMatrix)
        return true
    }

    /**
     * Mirrors PreviewView's FILL_CENTER: scale until the image covers the view, centre it, and
     * let the overflow fall outside the bounds. Results land in [mapping] as scale, dx, dy.
     */
    private fun computeMapping() {
        val scale = maxOf(width.toFloat() / imageWidth, height.toFloat() / imageHeight)
        mapping[0] = scale
        mapping[1] = (width - imageWidth * scale) / 2f
        mapping[2] = (height - imageHeight * scale) / 2f
    }

    private fun viewX(p: FloatArray, index: Int) = p[index * STRIDE] * imageWidth * mapping[0] + mapping[1]
    private fun viewY(p: FloatArray, index: Int) = p[index * STRIDE + 1] * imageHeight * mapping[0] + mapping[2]
    private fun visible(p: FloatArray, index: Int) =
        index >= 0 && index * STRIDE + 2 < p.size && p[index * STRIDE + 2] >= MIN_VISIBILITY

    /**
     * Draws the garment. Returns whether anything was drawn.
     *
     * Four passes: every limb outline, then the torso, then every limb fill, then every seam. A
     * limb's outline is its silhouette grown by a few px, so whatever is drawn over it covers it —
     * the torso where a sleeve overlaps the chest, its own fill everywhere inside. What survives is
     * an outline only where a limb is the garment's outer edge. Drawing each limb with its own
     * outline, on top of the torso, put a line across the chest wherever a sleeve overlapped it.
     *
     * That rule cannot tell the chest from the underarm, so it also hid the one edge that should
     * show over the body: a long sleeve's inside edge below the armpit, which made the sleeve fuse
     * into the shirt like a batwing. Seams put exactly that edge back, drawn last.
     *
     * The torso uses M3's affine: both shoulders and the hip midpoint define the triangle, so the
     * garment scales with shoulder width and torso length independently. Each other layer is a
     * segment that follows its own two landmarks — see [segmentMatrix] — and falls back to its
     * parent's transform when those landmarks are not visible, which leaves the part in its rest
     * position on the body instead of letting it vanish.
     *
     * The whole garment still needs the torso's four landmarks. Without them there is nothing to
     * hang the limbs from. Trousers go to [drawTrousers] instead, which needs only the hips.
     */
    private fun drawGarment(canvas: Canvas, p: FloatArray, worn: Worn): Boolean {
        worn.trousers?.let { return drawTrousers(canvas, p, it) }
        if (!visible(p, LEFT_SHOULDER) || !visible(p, RIGHT_SHOULDER) ||
            !visible(p, LEFT_HIP) || !visible(p, RIGHT_HIP)) return false
        worn.meshed?.let { return drawMeshGarment(canvas, p, it) }

        val r = worn.layered ?: return false
        val handedness = computeTorso(p, r.leftShoulder ?: return false, r.rightShoulder ?: return false,
            r.leftHip ?: return false, r.rightHip ?: return false)
        if (handedness == 0f) return false

        for (i in 0 until r.count) {
            val m = r.matrices[i]
            if (r.isTorso[i]) {
                m.set(torsoMatrix)
            } else if (!segmentMatrix(r.fromAnchor[i], r.toAnchor[i], r.fromLandmark[i],
                    r.toLandmark[i], p, handedness, m)) {
                val parent = r.parentIndex[i]
                m.set(if (parent >= 0) r.matrices[parent] else torsoMatrix)
            }
        }

        for (i in 0 until r.count) {
            if (!r.isTorso[i]) r.outlines[i]?.let { canvas.drawBitmap(it, r.matrices[i], garmentPaint) }
        }
        for (i in 0 until r.count) {
            if (r.isTorso[i]) canvas.drawBitmap(r.fills[i], r.matrices[i], garmentPaint)
        }
        for (i in 0 until r.count) {
            if (!r.isTorso[i]) canvas.drawBitmap(r.fills[i], r.matrices[i], garmentPaint)
        }
        for (i in 0 until r.count) {
            r.seams[i]?.let { canvas.drawBitmap(it, r.matrices[i], garmentPaint) }
        }
        return true
    }

    /**
     * Sets [torsoMatrix] and [torsoInverse] from the torso anchors: both shoulders and the hip
     * midpoint, so the garment scales with shoulder width and torso length independently (M3).
     *
     * Returns the transform's handedness — -1 if it mirrors the artwork, which every limb must then
     * copy or a sleeve's outer edge would end up facing the body — or 0 if it is unusable.
     */
    private fun computeTorso(p: FloatArray, ls: Anchor, rs: Anchor, lh: Anchor, rh: Anchor): Float {
        src[0] = ls.x; src[1] = ls.y
        src[2] = rs.x; src[3] = rs.y
        src[4] = (lh.x + rh.x) / 2f; src[5] = (lh.y + rh.y) / 2f

        dst[0] = viewX(p, LEFT_SHOULDER); dst[1] = viewY(p, LEFT_SHOULDER)
        dst[2] = viewX(p, RIGHT_SHOULDER); dst[3] = viewY(p, RIGHT_SHOULDER)
        dst[4] = (viewX(p, LEFT_HIP) + viewX(p, RIGHT_HIP)) / 2f
        dst[5] = (viewY(p, LEFT_HIP) + viewY(p, RIGHT_HIP)) / 2f

        if (!torsoMatrix.setPolyToPoly(src, 0, dst, 0, 3)) return 0f
        if (!torsoMatrix.invert(torsoInverse)) return 0f

        torsoMatrix.getValues(matrixValues)
        val det = matrixValues[Matrix.MSCALE_X] * matrixValues[Matrix.MSCALE_Y] -
            matrixValues[Matrix.MSKEW_X] * matrixValues[Matrix.MSKEW_Y]
        return if (det >= 0f) 1f else -1f
    }

    /**
     * Draws a real garment photo bent over its grid (M3.6). Returns whether anything was drawn.
     *
     * Three bones — torso, left sleeve, right sleeve — are computed exactly as for layered
     * garments. Each grid vertex then moves to the weighted blend of where each bone would put it,
     * so the photo bends smoothly across the armhole instead of breaking at a cut. A sleeve whose
     * elbow is out of sight simply follows the torso, leaving it at rest on the body.
     *
     * Two passes: the whole garment, then only its sleeves again on top. A bent mesh draws its
     * triangles in grid order, so without the second pass the body could land over a sleeve
     * wherever the two overlap.
     *
     * Mirrors render() in tools/preview_mesh.py. Change the two together.
     */
    private fun drawMeshGarment(canvas: Canvas, p: FloatArray, m: MeshRender): Boolean {
        val handedness = computeTorso(p, m.leftShoulder, m.rightShoulder, m.leftHip, m.rightHip)
        if (handedness == 0f) return false

        m.bones[0].set(torsoMatrix)
        val leftPlaced = m.leftSleeve?.let {
            sleeveMatrix(it, LEFT_SHOULDER, LEFT_ELBOW, p, handedness, m.sleeveAllowance, m.bones[1])
        } ?: segmentMatrix(m.leftShoulder, m.leftElbow, LEFT_SHOULDER, LEFT_ELBOW, p, handedness, m.bones[1])
        if (!leftPlaced) m.bones[1].set(torsoMatrix)
        val rightPlaced = m.rightSleeve?.let {
            sleeveMatrix(it, RIGHT_SHOULDER, RIGHT_ELBOW, p, handedness, m.sleeveAllowance, m.bones[2])
        } ?: segmentMatrix(m.rightShoulder, m.rightElbow, RIGHT_SHOULDER, RIGHT_ELBOW, p, handedness, m.bones[2])
        if (!rightPlaced) m.bones[2].set(torsoMatrix)

        fitToBody(m)
        m.mesh.widen(m.easeAmount, m.widened)
        for (k in 0 until GarmentMesh.BONES) m.bones[k].mapPoints(m.placed[k], m.widened)

        // The torso first, lifted with both armpits; then each sleeve, whose points on the body
        // side of its armhole follow that lifted torso - see GarmentMesh.BODY_SIDE.
        val torso = m.placed[0]
        for (side in 0..1) {
            m.fitAmount[side] = fitSleeve(m, side, p, handedness)
            if (m.fitAmount[side] <= 0f) continue
            val lift = m.mesh.armpitLift[side]
            val mx = m.armpitMove[side * 2]
            val my = m.armpitMove[side * 2 + 1]
            for (v in 0 until m.mesh.vertexCount) {
                torso[v * 2] += lift[v] * mx
                torso[v * 2 + 1] += lift[v] * my
            }
        }
        for (side in 0..1) {
            val t = m.fitAmount[side]
            if (t <= 0f) continue
            val placed = m.placed[side + 1]
            val u = m.mesh.sleeveU[side]
            val vv = m.mesh.sleeveV[side]
            val body = m.mesh.bodySide[side]
            val c = m.cornerTargets
            val o = side * 8
            for (v in 0 until m.mesh.vertexCount) {
                val a = (1 - u[v]) * (1 - vv[v])
                val b = u[v] * (1 - vv[v])
                val d = u[v] * vv[v]
                val e = (1 - u[v]) * vv[v]
                val x = v * 2
                val fx = a * c[o] + b * c[o + 2] + d * c[o + 4] + e * c[o + 6]
                val fy = a * c[o + 1] + b * c[o + 3] + d * c[o + 5] + e * c[o + 7]
                val tx = fx + body[v] * (torso[x] - fx)
                val ty = fy + body[v] * (torso[x + 1] - fy)
                placed[x] += t * (tx - placed[x])
                placed[x + 1] += t * (ty - placed[x + 1])
            }
        }

        val w = m.mesh.weights
        val t = m.placed[0]
        val l = m.placed[1]
        val r = m.placed[2]
        val out = m.vertices
        for (v in 0 until m.mesh.vertexCount) {
            val wt = w[v * 3]
            val wl = w[v * 3 + 1]
            val wr = w[v * 3 + 2]
            val x = v * 2
            out[x] = wt * t[x] + wl * l[x] + wr * r[x]
            out[x + 1] = wt * t[x + 1] + wl * l[x + 1] + wr * r[x + 1]
        }

        val cols = m.mesh.cols
        val rows = m.mesh.rows
        canvas.drawBitmapMesh(m.front, cols, rows, out, 0, null, 0, garmentPaint)
        canvas.drawBitmapMesh(m.sleeves, cols, rows, out, 0, null, 0, garmentPaint)
        return true
    }

    /**
     * Draws a pair of trousers bent over its grid (M4). Returns whether anything was drawn.
     *
     * Three bones. The pelvis, from [computePelvis], carries the waistband. Each leg swings from
     * its hip to its ankle: as long as the wearer's leg and as wide as the pelvis is across it,
     * through [segmentMatrix] as for a shirt's sleeves. The grid is first widened to the wearer by
     * [fitTrousers]. Straight legs for now: the knee only
     * points the leg when the ankle is out of sight, and a leg with neither in sight hangs from
     * the pelvis at rest. Every grid vertex then moves to the weighted blend of where each bone
     * would put it, as for a shirt.
     *
     * Needs both hips. Mirrors place() in tools/preview_trousers.py. Change the two together.
     */
    private fun drawTrousers(canvas: Canvas, p: FloatArray, t: TrouserRender): Boolean {
        if (!visible(p, LEFT_HIP) || !visible(p, RIGHT_HIP)) return false
        val handedness = computePelvis(p, t)
        if (handedness == 0f) return false

        t.bones[0].set(pelvisMatrix)
        if (!legMatrix(t.leftHip, t.leftKnee, t.leftAnkle, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE,
                p, handedness, t.bones[1])) t.bones[1].set(pelvisMatrix)
        if (!legMatrix(t.rightHip, t.rightKnee, t.rightAnkle, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
                p, handedness, t.bones[2])) t.bones[2].set(pelvisMatrix)
        fitTrousers(t)
        t.mesh.widen(t.outer, t.inner, t.widened)
        for (k in 0 until TrouserMesh.BONES) t.bones[k].mapPoints(t.placed[k], t.widened)

        val w = t.mesh.weights
        val pelvis = t.placed[0]
        val l = t.placed[1]
        val r = t.placed[2]
        val out = t.vertices
        for (v in 0 until t.mesh.vertexCount) {
            val wp = w[v * 3]
            val wl = w[v * 3 + 1]
            val wr = w[v * 3 + 2]
            val x = v * 2
            out[x] = wp * pelvis[x] + wl * l[x] + wr * r[x]
            out[x + 1] = wp * pelvis[x + 1] + wl * l[x + 1] + wr * r[x + 1]
        }
        canvas.drawBitmapMesh(t.front, t.mesh.cols, t.mesh.rows, out, 0, null, 0, garmentPaint)
        return true
    }

    /**
     * Sets [TrouserRender.outer] and [TrouserRender.inner], the widening per leg per grid row, for
     * the current [bottomFitMode]. Mirrors fit() in tools/preview_trousers.py. Change them together.
     *
     * - [FitMode.CHEST] fits by the hips. Each side of the garment at its hip anchors' row is made
     *   as wide as that side of the wearer at the hip line - the median of the measured levels
     *   from [HIP_FROM] to [HIP_TO] - times the fit allowance, and every row gets that one
     *   widening, so the legs keep the garment's own shape. Left and right are fitted apart: the
     *   tester's body reached further right of the hip joints than left, and a fit centred on the
     *   joints left his own trousers showing on that side.
     * - [FitMode.ROWS] fits row by row. The seat: each row to the wearer's width at the height it
     *   lands on, along the shoulder-to-hip axis. The legs: each to its measured thickness at that
     *   point, outer and inner sides apart; rows of the seat above the first leg row take that
     *   row's. The two blend down the thigh along [TrouserMesh.legBlend]. Whatever is not
     *   measured falls back to the hips' fit.
     * - [FitMode.OFF] widens nothing.
     *
     * With nothing measured at the hips this frame the last fit is kept. Needs the bones set this
     * frame.
     */
    private fun fitTrousers(t: TrouserRender) {
        if (bottomFitMode == FitMode.OFF) {
            for (s in 0..1) {
                t.outer[s].fill(0f)
                t.inner[s].fill(0f)
            }
            trouserEase = 0f
            return
        }
        val body = bodyWidths ?: return
        t.bones[0].getValues(matrixValues)
        val rowScale = hypot(matrixValues[Matrix.MSCALE_X], matrixValues[Matrix.MSKEW_Y])
        if (rowScale <= 0f) return
        val mesh = t.mesh
        val levels = BodyWidthMeter.LEVELS
        levelHalves(body, t.levelHalves)

        for (s in 0..1) {
            var n = 0
            for (k in levels.indices) {
                val half = t.levelHalves[s][k]
                if (levels[k] in HIP_FROM..HIP_TO && !half.isNaN()) t.hipScratch[n++] = half
            }
            val garmentHalf = mesh.halfWidth[s][mesh.hipRow]
            t.hipEase[s] = if (n == 0 || garmentHalf < 1f) Float.NaN else {
                java.util.Arrays.sort(t.hipScratch, 0, n)
                (t.hipScratch[n / 2] * t.allowance / (garmentHalf * rowScale) - 1f).coerceIn(MIN_EASE, MAX_EASE)
            }
        }
        // A side with nothing measured takes the other side's.
        if (t.hipEase[0].isNaN() && t.hipEase[1].isNaN()) return
        if (t.hipEase[0].isNaN()) t.hipEase[0] = t.hipEase[1]
        if (t.hipEase[1].isNaN()) t.hipEase[1] = t.hipEase[0]

        if (bottomFitMode == FitMode.CHEST) {
            for (s in 0..1) {
                t.outer[s].fill(t.hipEase[s])
                t.inner[s].fill(t.hipEase[s])
            }
            trouserEase = (t.hipEase[0] + t.hipEase[1]) / 2f
            return
        }

        // The wearer's shoulder-to-hip axis, from the first measuring level to the hip line's.
        val stride = BodyWidthMeter.STRIDE
        val hipLevel = levels.indexOfFirst { it >= 1f }
        val sx = imageWidth * mapping[0]
        val sy = imageHeight * mapping[0]
        val topX = body[0] * sx + mapping[1]
        val topY = body[1] * sy + mapping[2]
        val axisX = body[hipLevel * stride] * sx + mapping[1] - topX
        val axisY = body[hipLevel * stride + 1] * sy + mapping[2] - topY
        val axisSq = axisX * axisX + axisY * axisY
        if (axisSq < 1f) return

        val legs = legWidths
        for (s in 0..1) {
            for (i in BodyWidthMeter.LEG_SAMPLES.indices) {
                val p = s * BodyWidthMeter.LEG_STRIDE + i * BodyWidthMeter.LEG_SAMPLE_STRIDE
                t.legOuterView[s][i] = if (legs == null || p + 1 >= legs.size) Float.NaN else legs[p] * sx
                t.legInnerView[s][i] = if (legs == null || p + 1 >= legs.size) Float.NaN else legs[p + 1] * sx
            }
        }

        var sum = 0f
        var count = 0
        val stridePoints = (mesh.cols + 1) * 2
        for (s in 0..1) {
            vector[0] = t.acrossX[s]
            vector[1] = t.acrossY[s]
            t.bones[s + 1].mapVectors(vector)
            val legScale = hypot(vector[0], vector[1])
            for (r in 0..mesh.rows) {
                point[0] = mesh.centreX
                point[1] = mesh.source[r * stridePoints + 1]
                t.bones[0].mapPoints(point)
                val level = levels[0] + (levels[hipLevel] - levels[0]) *
                    ((point[0] - topX) * axisX + (point[1] - topY) * axisY) / axisSq
                val bodyHalf = widthAt(t.levelHalves[s], level)
                val garmentHalf = mesh.halfWidth[s][r]
                val seat = if (bodyHalf.isNaN() || garmentHalf < 1f) t.hipEase[s]
                    else (bodyHalf * t.allowance / (garmentHalf * rowScale) - 1f).coerceIn(MIN_EASE, MAX_EASE)
                val lr = maxOf(r, mesh.firstLegRow[s])
                val legT = mesh.legT[s][lr]
                val legOut = legEase(t.legOuterView[s], legT, mesh.legOuter[s][lr], legScale, t.allowance, t.hipEase[s])
                val legIn = legEase(t.legInnerView[s], legT, mesh.legInner[s][lr], legScale, t.allowance, t.hipEase[s])
                val f = mesh.legBlend[s][r]
                t.outer[s][r] = (1f - f) * seat + f * legOut
                t.inner[s][r] = (1f - f) * seat + f * legIn
                if (garmentHalf >= 1f) {
                    sum += t.outer[s][r]
                    count++
                }
            }
        }
        trouserEase = if (count > 0) sum / count else Float.NaN
    }

    /**
     * One side of a leg's widening at a row: the wearer's half-thickness there, from the samples in
     * [view], over the garment's [garmentHalf] across the leg on screen; [fallback] where either is
     * unknown.
     */
    private fun legEase(
        view: FloatArray, legT: Float, garmentHalf: Float, legScale: Float, allowance: Float, fallback: Float
    ): Float {
        val bodyHalf = widthAt(view, legT, BodyWidthMeter.LEG_SAMPLES)
        if (bodyHalf.isNaN() || garmentHalf < 1f || legScale <= 0f) return fallback
        return (bodyHalf * allowance / (garmentHalf * legScale) - 1f).coerceIn(MIN_EASE, MAX_EASE)
    }

    /** Each measured level's left and right half-widths, in view px; NaN where unknown. */
    private fun levelHalves(body: FloatArray, out: Array<FloatArray>) {
        val sx = imageWidth * mapping[0]
        val sy = imageHeight * mapping[0]
        val stride = BodyWidthMeter.STRIDE
        for (k in BodyWidthMeter.LEVELS.indices) {
            val o = k * stride
            if (o + stride > body.size) {
                out[0][k] = Float.NaN
                out[1][k] = Float.NaN
                continue
            }
            val quality = body[o + 6].toInt()
            val cx = body[o] * sx
            val cy = body[o + 1] * sy
            out[0][k] = if (quality / 3 != BodyWidthMeter.NONE) {
                hypot(body[o + 2] * sx - cx, body[o + 3] * sy - cy)
            } else Float.NaN
            out[1][k] = if (quality % 3 != BodyWidthMeter.NONE) {
                hypot(body[o + 4] * sx - cx, body[o + 5] * sy - cy)
            } else Float.NaN
        }
    }

    /** One leg's bone: hip to ankle, or hip to knee when the ankle is out of sight. */
    private fun legMatrix(
        hip: Anchor, knee: Anchor, ankle: Anchor, hipAt: Int, kneeAt: Int, ankleAt: Int,
        p: FloatArray, handedness: Float, out: Matrix
    ): Boolean = segmentMatrix(hip, ankle, hipAt, ankleAt, p, handedness, out, pelvisInverse) ||
        segmentMatrix(hip, knee, hipAt, kneeAt, p, handedness, out, pelvisInverse)

    /**
     * Sets [pelvisMatrix] and [pelvisInverse] for a pair of trousers. Returns the transform's
     * handedness, as [computeTorso] does, or 0 if it is unusable.
     *
     * Three points, as M3's torso has: the garment's hip anchors go onto the wearer's hips, and
     * the point one garment leg length below the anchors' midpoint goes onto the point one of the
     * wearer's leg lengths below the hips'. So the hips' spacing sets the scale across, the legs
     * set it down, and the waistband and seat stay in proportion with the legs drawn below them.
     * "Below" is square to the hip line, on the image's lower side.
     *
     * The wearer's leg length is the mean of the legs in sight: hip to ankle, or hip to knee
     * scaled up by the garment's own leg-to-thigh proportion. With no leg in sight, the hips'
     * spacing sets the scale both ways.
     */
    private fun computePelvis(p: FloatArray, t: TrouserRender): Float {
        val lx = viewX(p, LEFT_HIP)
        val ly = viewY(p, LEFT_HIP)
        val rx = viewX(p, RIGHT_HIP)
        val ry = viewY(p, RIGHT_HIP)
        val span = hypot(rx - lx, ry - ly)
        if (span < 1f || t.hipSpan < 1f) return 0f

        var total = 0f
        var legs = 0
        for (side in 0..1) {
            val hipAt = if (side == 0) LEFT_HIP else RIGHT_HIP
            val kneeAt = if (side == 0) LEFT_KNEE else RIGHT_KNEE
            val ankleAt = if (side == 0) LEFT_ANKLE else RIGHT_ANKLE
            val hx = viewX(p, hipAt)
            val hy = viewY(p, hipAt)
            if (visible(p, ankleAt)) {
                total += hypot(viewX(p, ankleAt) - hx, viewY(p, ankleAt) - hy)
                legs++
            } else if (visible(p, kneeAt)) {
                total += hypot(viewX(p, kneeAt) - hx, viewY(p, kneeAt) - hy) * t.legPerThigh[side]
                legs++
            }
        }
        val length = if (legs > 0) total / legs else t.legLength * span / t.hipSpan

        // Square to the hip line, (-dy, dx) / |d|, turned to point down the image.
        val ax = t.rightHip.x - t.leftHip.x
        val ay = t.rightHip.y - t.leftHip.y
        val artDown = if (ax >= 0f) 1f else -1f
        src[0] = t.leftHip.x; src[1] = t.leftHip.y
        src[2] = t.rightHip.x; src[3] = t.rightHip.y
        src[4] = (t.leftHip.x + t.rightHip.x) / 2f - artDown * ay / t.hipSpan * t.legLength
        src[5] = (t.leftHip.y + t.rightHip.y) / 2f + artDown * ax / t.hipSpan * t.legLength

        val viewDown = if (rx - lx >= 0f) 1f else -1f
        dst[0] = lx; dst[1] = ly
        dst[2] = rx; dst[3] = ry
        dst[4] = (lx + rx) / 2f - viewDown * (ry - ly) / span * length
        dst[5] = (ly + ry) / 2f + viewDown * (rx - lx) / span * length

        if (!pelvisMatrix.setPolyToPoly(src, 0, dst, 0, 3)) return 0f
        if (!pelvisMatrix.invert(pelvisInverse)) return 0f

        pelvisMatrix.getValues(matrixValues)
        val det = matrixValues[Matrix.MSCALE_X] * matrixValues[Matrix.MSCALE_Y] -
            matrixValues[Matrix.MSKEW_X] * matrixValues[Matrix.MSKEW_Y]
        return if (det >= 0f) 1f else -1f
    }

    /**
     * Sets [MeshRender.easeAmount] so that, row by row, the garment's body is as wide as the
     * wearer's torso - measured live by [BodyWidthMeter] - times the garment's fit allowance.
     *
     * Each grid row sits at a height on the torso, [GarmentMesh.rowT]; the wearer's width there is
     * interpolated between the measured levels around it. The garment's own width at that row,
     * [GarmentMesh.torsoWidth], becomes its on-screen width through the torso transform's scale
     * along the row. The widening is whatever makes the two match.
     *
     * With no measurement this frame the amounts are left as they were, so the garment keeps its
     * last fit rather than jumping. Requires [computeTorso] to have run this frame.
     */
    private fun fitToBody(m: MeshRender) {
        if (topFitMode == FitMode.OFF) {
            m.easeAmount.fill(0f)
            fittedEase = 0f
            return
        }
        val body = bodyWidths ?: return
        val levels = BodyWidthMeter.LEVELS
        val stride = BodyWidthMeter.STRIDE
        val sx = imageWidth * mapping[0]
        val sy = imageHeight * mapping[0]
        var known = 0
        for (k in levels.indices) {
            val o = k * stride
            if (o + stride > body.size) {
                m.levelWidths[k] = Float.NaN
                continue
            }
            val quality = body[o + 6].toInt()
            val cx = body[o] * sx
            val cy = body[o + 1] * sy
            val leftHalf = if (quality / 3 != BodyWidthMeter.NONE) {
                hypot(body[o + 2] * sx - cx, body[o + 3] * sy - cy)
            } else Float.NaN
            val rightHalf = if (quality % 3 != BodyWidthMeter.NONE) {
                hypot(body[o + 4] * sx - cx, body[o + 5] * sy - cy)
            } else Float.NaN
            m.levelWidths[k] = when {
                !leftHalf.isNaN() && !rightHalf.isNaN() -> leftHalf + rightHalf
                !leftHalf.isNaN() -> 2f * leftHalf
                !rightHalf.isNaN() -> 2f * rightHalf
                else -> Float.NaN
            }
            if (!m.levelWidths[k].isNaN()) known++
        }
        if (known == 0) return

        // On-screen px per artwork px along a garment row.
        torsoMatrix.getValues(matrixValues)
        val rowScale = hypot(matrixValues[Matrix.MSCALE_X], matrixValues[Matrix.MSKEW_Y])
        if (rowScale <= 0f) return

        if (topFitMode == FitMode.CHEST) {
            fitToChest(m, rowScale)
            return
        }

        var sum = 0f
        var count = 0
        for (r in 0..m.mesh.rows) {
            val garmentWidth = m.mesh.torsoWidth[r]
            val bodyWidth = widthAt(m.levelWidths, m.mesh.rowT[r])
            m.easeAmount[r] = if (garmentWidth < 1f || bodyWidth.isNaN()) 0f
            else (bodyWidth * m.allowance / (garmentWidth * rowScale) - 1f).coerceIn(MIN_EASE, MAX_EASE)
            if (m.mesh.rowT[r] in 0.5f..1.1f) {
                sum += m.easeAmount[r]
                count++
            }
        }
        fittedEase = if (count > 0) sum / count else Float.NaN
    }

    /**
     * [FitMode.CHEST]: one widening for every row, making the garment's chest as wide as the
     * wearer's - the median of the measured levels in the chest band below the armpit - times
     * the fit allowance. With no level measured in the band, the last fit is kept.
     */
    private fun fitToChest(m: MeshRender, rowScale: Float) {
        val levels = BodyWidthMeter.LEVELS
        val from = m.mesh.armpitT + GarmentMesh.CHEST_START
        val to = from + GarmentMesh.CHEST_SPAN
        var n = 0
        for (k in levels.indices) {
            if (levels[k] in from..to && !m.levelWidths[k].isNaN()) m.chestScratch[n++] = m.levelWidths[k]
        }
        if (n == 0 || m.mesh.chestWidth < 1f) return
        java.util.Arrays.sort(m.chestScratch, 0, n)
        val bodyChest = m.chestScratch[n / 2]
        val e = (bodyChest * m.allowance / (m.mesh.chestWidth * rowScale) - 1f).coerceIn(MIN_EASE, MAX_EASE)
        for (r in 0..m.mesh.rows) {
            m.easeAmount[r] = if (m.mesh.torsoWidth[r] < 1f) 0f else e
        }
        fittedEase = e
    }

    /**
     * The wearer's torso width at height [t], interpolated between the nearest measured levels
     * either side, or the nearest one beyond the measured range; NaN if none is measured.
     * [positions] says where each entry of [widths] was measured: the torso's levels unless given,
     * or [BodyWidthMeter.LEG_SAMPLES] for a leg.
     */
    private fun widthAt(widths: FloatArray, t: Float, positions: FloatArray = BodyWidthMeter.LEVELS): Float {
        val levels = positions
        var below = -1
        var above = -1
        for (k in levels.indices) {
            if (widths[k].isNaN()) continue
            if (levels[k] <= t) below = k
            if (levels[k] >= t && above < 0) above = k
        }
        return when {
            below >= 0 && above >= 0 && above != below -> {
                val f = (t - levels[below]) / (levels[above] - levels[below])
                widths[below] + f * (widths[above] - widths[below])
            }
            below >= 0 -> widths[below]
            above >= 0 -> widths[above]
            else -> Float.NaN
        }
    }

    /**
     * Fits one sleeve ([side] 0 = left) to the arm by its four corners, for an arm raised past the
     * sleeve's own angle in the photo. Returns how far the fit applies, 0..1 - 0 leaves the sleeve
     * where [sleeveMatrix] put it - and fills its entries of [MeshRender.cornerTargets] and
     * [MeshRender.armpitMove].
     *
     * Turning the photo's hanging sleeve as a whole about the shoulder kept it as tall as it hangs:
     * with the arm raised it stood above the arm like a box, and its underside hung below the arm
     * as a dark flap. Instead each corner is placed on the body. The shoulder tip stays where the
     * torso puts it, so the seam cannot open. The cuff's two ends sit across the arm at its width
     * times the sleeve allowance. The armpit slides straight up the torso's side, from where the
     * torso puts it to [ARMPIT_TUCK] below the arm's line, and the torso nearest it follows, so the underarm
     * fabric is taken up into the armpit rather than left hanging. Every vertex keeps its place
     * between the four corners, so the photo's texture and shading come with it.
     *
     * The fit fades in over [RISE_DEG] above the sleeve's angle in the photo: at rest, nothing
     * changes. It is skipped when the corners would land twisted - an arm folded across the body.
     *
     * Mirrors fit_sleeve() in tools/preview_mesh.py. Change the two together.
     */
    private fun fitSleeve(m: MeshRender, side: Int, p: FloatArray, handedness: Float): Float {
        val k = m.mesh.sleeveCorners[side] ?: return 0f
        val frame = (if (side == 0) m.leftSleeve else m.rightSleeve) ?: return 0f
        val joint = if (side == 0) LEFT_SHOULDER else RIGHT_SHOULDER
        val elbow = if (side == 0) LEFT_ELBOW else RIGHT_ELBOW
        if (!visible(p, joint) || !visible(p, elbow)) return 0f

        val jx = viewX(p, joint)
        val jy = viewY(p, joint)
        val length = hypot(viewX(p, elbow) - jx, viewY(p, elbow) - jy)
        if (length < 1f) return 0f
        val bx = (viewX(p, elbow) - jx) / length
        val by = (viewY(p, elbow) - jy) / length
        val downX = (viewX(p, LEFT_HIP) + viewX(p, RIGHT_HIP) - viewX(p, LEFT_SHOULDER) - viewX(p, RIGHT_SHOULDER)) / 2f
        val downY = (viewY(p, LEFT_HIP) + viewY(p, RIGHT_HIP) - viewY(p, LEFT_SHOULDER) - viewY(p, RIGHT_SHOULDER)) / 2f
        val rise = ((degreesBetween(bx, by, downX, downY) - m.mesh.sleeveAngle[side]) / RISE_DEG).coerceIn(0f, 1f)
        val t = rise * rise * (3f - 2f * rise)
        if (t <= 0f) return 0f

        // Shoulder tip and armpit where the widened torso puts them.
        val s = m.scratch
        s[0] = m.mesh.widenOnBoundaryX(k[0], k[1], m.easeAmount); s[1] = k[1]
        s[2] = m.mesh.widenOnBoundaryX(k[6], k[7], m.easeAmount); s[3] = k[7]
        torsoMatrix.mapPoints(s)
        val a0x = s[2]
        val a0y = s[3]

        // Across the arm, pointing away from the armpit.
        var cx = -by
        var cy = bx
        if ((a0x - jx) * cx + (a0y - jy) * cy > 0f) {
            cx = -cx
            cy = -cy
        }
        val joints = hypot(viewX(p, RIGHT_SHOULDER) - viewX(p, LEFT_SHOULDER),
            viewY(p, RIGHT_SHOULDER) - viewY(p, LEFT_SHOULDER))
        // The sleeve spans the arm's measured edges, or an arm width assumed from the shoulders
        // when the arm has not been measured.
        val armSx = imageWidth * mapping[0]
        val arms = armWidths
        val ao = side * BodyWidthMeter.ARM_STRIDE
        val armOuter = if (arms != null && ao + 1 < arms.size) arms[ao] * armSx else Float.NaN
        val armInner = if (arms != null && ao + 1 < arms.size) arms[ao + 1] * armSx else Float.NaN
        val measured = !armOuter.isNaN() && !armInner.isNaN()
        val half = 0.5f * ARM_WIDTH_PER_JOINT * joints * m.sleeveAllowance
        val outerHalf = if (measured) armOuter * m.measuredSleeveAllowance else half
        val innerHalf = if (measured) armInner * m.measuredSleeveAllowance else half
        val along = frame.cuffAlong * length / frame.length
        val mx = jx + bx * along
        val my = jy + by * along

        val c = m.cornerTargets
        val b = side * 8
        c[b] = s[0]; c[b + 1] = s[1]
        c[b + 2] = mx + cx * outerHalf; c[b + 3] = my + cy * outerHalf
        c[b + 4] = mx - cx * innerHalf; c[b + 5] = my - cy * innerHalf
        // The armpit rises straight up the torso's side, towards the tuck point only along the
        // body's own up-down line. Moving it the whole way to the tuck point - up and inwards,
        // under the shoulder joint - pulled the shirt's side in with it, and on device the shirt
        // pinched in below each raised arm and showed the wearer's own clothes there.
        //
        // With the arm measured, it rises to where the sleeve's underside - the arm's inner edge,
        // running along the arm - meets the torso's side, so the sleeve's bottom edge follows the
        // arm into the armpit. Rising a fixed distance instead left the arm's underside uncovered
        // near the armpit at 45 degrees, and at 90 tucked the root below the arm so the sleeve
        // narrowed from root to cuff and looked bent into the shirt.
        val downLength = hypot(downX, downY)
        val ux = downX / downLength
        val uy = downY / downLength
        var climb = (jx - cx * ARMPIT_TUCK * joints - a0x) * ux + (jy - cy * ARMPIT_TUCK * joints - a0y) * uy
        if (measured) {
            // Solve J - c * innerHalf + b * s = A0 + u * k for k, the distance along the torso's side.
            val rx = a0x - (jx - cx * innerHalf)
            val ry = a0y - (jy - cy * innerHalf)
            val det = -bx * uy + ux * by
            climb = if (abs(det) < 1e-3f) 0f else ((bx * ry - by * rx) / det).coerceAtMost(0f)
            // Never above the shoulder joint's height.
            climb = climb.coerceAtLeast((jx - a0x) * ux + (jy - a0y) * uy)
        }
        c[b + 6] = a0x + t * climb * ux
        c[b + 7] = a0y + t * climb * uy

        val want = m.mesh.sleeveWinding[side] * handedness
        for (i in 0 until 4) {
            val o = b + i * 2
            val n = b + ((i + 1) % 4) * 2
            val q = b + ((i + 3) % 4) * 2
            val cross = (c[n] - c[o]) * (c[q + 1] - c[o + 1]) - (c[n + 1] - c[o + 1]) * (c[q] - c[o])
            if (cross * want <= 0f) return 0f
        }
        m.armpitMove[side * 2] = c[b + 6] - a0x
        m.armpitMove[side * 2 + 1] = c[b + 7] - a0y
        return t
    }

    /**
     * Places one sleeve of a real garment on the arm: the sleeve's own centre line along the arm,
     * shoulder joint to elbow, and its width across that line squeezed or stretched to the arm's
     * width times the garment's [sleeveAllowance]. Returns false when the arm is not visible.
     *
     * Rotating the whole sleeve about the shoulder joint, along the joint-to-elbow anchors, swung
     * a wide flat sleeve's underarm half across the chest when the arm hung down: on a flat-laid
     * shirt over half its grid points landed more than 20 px inside the arm, some nearly 300 px.
     * A photographed sleeve is a fixed width whatever arm it lands on; worn, it wraps that arm.
     *
     * Mirrors sleeve_matrix() in tools/preview_mesh.py. Change the two together.
     */
    private fun sleeveMatrix(
        f: SleeveFrame, from: Int, to: Int, p: FloatArray, handedness: Float, allowance: Float,
        out: Matrix
    ): Boolean {
        if (!visible(p, from) || !visible(p, to)) return false
        val sx = viewX(p, from)
        val sy = viewY(p, from)
        val ex = viewX(p, to) - sx
        val ey = viewY(p, to) - sy
        val length = hypot(ex, ey)
        val joints = hypot(viewX(p, RIGHT_SHOULDER) - viewX(p, LEFT_SHOULDER),
            viewY(p, RIGHT_SHOULDER) - viewY(p, LEFT_SHOULDER))
        if (length < 1f || joints < 1f) return false
        val bx = ex / length
        val by = ey / length
        // Across the arm on screen, turned the same way the torso is.
        val cx = handedness * -by
        val cy = handedness * bx
        val across = ARM_WIDTH_PER_JOINT * joints * allowance / f.width

        src[0] = f.originX; src[1] = f.originY
        src[2] = f.originX + f.ux * f.length; src[3] = f.originY + f.uy * f.length
        src[4] = f.originX - f.uy * PERPENDICULAR_REACH; src[5] = f.originY + f.ux * PERPENDICULAR_REACH

        dst[0] = sx; dst[1] = sy
        dst[2] = sx + bx * length; dst[3] = sy + by * length
        dst[4] = sx + cx * PERPENDICULAR_REACH * across; dst[5] = sy + cy * PERPENDICULAR_REACH * across

        return out.setPolyToPoly(src, 0, dst, 0, 3)
    }

    /**
     * Builds the transform for one limb, from its anchors [a0] -> [a1] on the artwork to its
     * landmarks [from] -> [to] on the body. Returns false when either is not usable. [parentInverse]
     * is the inverse of the transform the limb hangs from: the torso's for a sleeve, the pelvis's
     * for a trouser leg.
     *
     * Length follows the limb, width follows the body. Along the axis from pivot to far end, the
     * part stretches to the wearer's actual segment length. Across it, the part scales with the
     * torso's stretch *in that same direction*: the torso transform is anisotropic — on the test
     * subject 2.1 screen px per artwork px across the shoulders but 1.3 down the torso — and a
     * sleeve's width runs horizontally when the arm hangs but vertically when it is raised. Using
     * the horizontal scale throughout made a raised sleeve ~60% too thick; reading the scale along
     * the actual perpendicular, k = 1 / |T⁻¹·d|, keeps it the thickness of the shirt it belongs to.
     *
     * The same step-for-step maths lives in tools/preview_garments.py, which is how poses were
     * checked before reaching a phone. Change the two together.
     */
    private fun segmentMatrix(
        a0: Anchor?, a1: Anchor?, from: Int, to: Int, p: FloatArray, handedness: Float, out: Matrix,
        parentInverse: Matrix = torsoInverse
    ): Boolean {
        if (a0 == null || a1 == null) return false
        if (!visible(p, from) || !visible(p, to)) return false

        val ax = a1.x - a0.x
        val ay = a1.y - a0.y
        val artLength = hypot(ax, ay)

        val v0x = viewX(p, from)
        val v0y = viewY(p, from)
        val vx = viewX(p, to) - v0x
        val vy = viewY(p, to) - v0y
        val viewLength = hypot(vx, vy)
        if (artLength < 1f || viewLength < 1f) return false

        // Unit perpendicular to the limb on screen, turned the same way the torso is.
        val dx = handedness * -vy / viewLength
        val dy = handedness * vx / viewLength

        vector[0] = dx
        vector[1] = dy
        parentInverse.mapVectors(vector)
        val inverseLength = hypot(vector[0], vector[1])
        if (inverseLength < 1e-6f) return false
        val widthScale = 1f / inverseLength

        src[0] = a0.x; src[1] = a0.y
        src[2] = a1.x; src[3] = a1.y
        src[4] = a0.x - ay / artLength * PERPENDICULAR_REACH
        src[5] = a0.y + ax / artLength * PERPENDICULAR_REACH

        dst[0] = v0x; dst[1] = v0y
        dst[2] = v0x + vx; dst[3] = v0y + vy
        dst[4] = v0x + dx * PERPENDICULAR_REACH * widthScale
        dst[5] = v0y + dy * PERPENDICULAR_REACH * widthScale

        return out.setPolyToPoly(src, 0, dst, 0, 3)
    }

    /**
     * Draws the skeleton. When a garment is on, it is dimmed rather than hidden — seeing the
     * joints through the shirt is how you tell a correct fit from a plausible-looking one.
     */
    private fun drawSkeleton(canvas: Canvas, p: FloatArray, dimmed: Boolean) {
        bonePaint.alpha = if (dimmed) 90 else 255
        jointPaint.alpha = if (dimmed) 120 else 255

        val scale = mapping[0]
        val dx = mapping[1]
        val dy = mapping[2]

        var n = 0
        var i = 0
        while (i < CONNECTIONS.size) {
            val a = CONNECTIONS[i]
            val b = CONNECTIONS[i + 1]
            i += 2
            if (!visible(p, a) || !visible(p, b)) continue

            segments[n++] = p[a * STRIDE] * imageWidth * scale + dx
            segments[n++] = p[a * STRIDE + 1] * imageHeight * scale + dy
            segments[n++] = p[b * STRIDE] * imageWidth * scale + dx
            segments[n++] = p[b * STRIDE + 1] * imageHeight * scale + dy
        }
        if (n > 0) canvas.drawLines(segments, 0, n, bonePaint)

        var j = 0
        while (j * STRIDE + 2 < p.size) {
            if (visible(p, j)) {
                canvas.drawCircle(viewX(p, j), viewY(p, j), jointRadius, jointPaint)
            }
            j++
        }
    }

    /**
     * Draws each measuring line, as two halves from the centre line out to the measured edges,
     * coloured by how each half was obtained. A debug view of [BodyWidthMeter], not part of the
     * try-on itself.
     */
    private fun drawArmWidths(canvas: Canvas, a: FloatArray) {
        val sx = imageWidth * mapping[0]
        val sy = imageHeight * mapping[0]
        for (side in 0..1) {
            for (i in BodyWidthMeter.ARM_SAMPLES.indices) {
                val p = side * BodyWidthMeter.ARM_STRIDE + 2 + i * 4
                if (p + 3 >= a.size || a[p].isNaN()) continue
                canvas.drawLine(a[p] * sx + mapping[1], a[p + 1] * sy + mapping[2],
                    a[p + 2] * sx + mapping[1], a[p + 3] * sy + mapping[2], measuredPaint)
            }
        }
    }

    /** Draws each leg measurement made this frame as a line across the leg, edge to edge. */
    private fun drawLegWidths(canvas: Canvas, legs: FloatArray) {
        val sx = imageWidth * mapping[0]
        val sy = imageHeight * mapping[0]
        for (side in 0..1) {
            for (i in BodyWidthMeter.LEG_SAMPLES.indices) {
                val p = side * BodyWidthMeter.LEG_STRIDE + i * BodyWidthMeter.LEG_SAMPLE_STRIDE + 2
                if (p + 3 >= legs.size || legs[p].isNaN()) continue
                canvas.drawLine(legs[p] * sx + mapping[1], legs[p + 1] * sy + mapping[2],
                    legs[p + 2] * sx + mapping[1], legs[p + 3] * sy + mapping[2], measuredPaint)
            }
        }
    }

    private fun drawBodyWidths(canvas: Canvas, w: FloatArray) {
        val sx = imageWidth * mapping[0]
        val sy = imageHeight * mapping[0]
        var o = 0
        while (o + BodyWidthMeter.STRIDE <= w.size) {
            val quality = w[o + 6].toInt()
            val cx = w[o] * sx + mapping[1]
            val cy = w[o + 1] * sy + mapping[2]
            drawHalfWidth(canvas, cx, cy, w[o + 2] * sx + mapping[1], w[o + 3] * sy + mapping[2], quality / 3)
            drawHalfWidth(canvas, cx, cy, w[o + 4] * sx + mapping[1], w[o + 5] * sy + mapping[2], quality % 3)
            o += BodyWidthMeter.STRIDE
        }
    }

    private fun drawHalfWidth(canvas: Canvas, cx: Float, cy: Float, ex: Float, ey: Float, quality: Int) {
        if (quality == BodyWidthMeter.NONE) return
        canvas.drawLine(cx, cy, ex, ey, if (quality == BodyWidthMeter.MEASURED) measuredPaint else filledPaint)
    }

    private fun drawGuide(canvas: Canvas) {
        val inset = width * 0.14f
        val vInset = height * 0.09f
        guide.set(inset, vInset, width - inset, height - vInset)
        val arm = minOf(guide.width(), guide.height()) * 0.08f

        canvas.drawLines(
            floatArrayOf(
                guide.left, guide.top, guide.left + arm, guide.top,
                guide.left, guide.top, guide.left, guide.top + arm,
                guide.right, guide.top, guide.right - arm, guide.top,
                guide.right, guide.top, guide.right, guide.top + arm,
                guide.left, guide.bottom, guide.left + arm, guide.bottom,
                guide.left, guide.bottom, guide.left, guide.bottom - arm,
                guide.right, guide.bottom, guide.right - arm, guide.bottom,
                guide.right, guide.bottom, guide.right, guide.bottom - arm
            ),
            bracketPaint
        )

        canvas.drawText(
            "STAND WITH FULL BODY IN FRAME",
            guide.centerX(),
            guide.bottom + 22f * density,
            captionPaint
        )
    }
}
