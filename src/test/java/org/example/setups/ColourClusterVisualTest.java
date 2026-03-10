package org.example.setups;
import org.example.OpenCvLoader;
import org.example.colour.SceneColourClusters;
import org.example.factories.BackgroundFactory;
import org.example.factories.BackgroundId;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.io.IOException;
import java.nio.file.*;
import java.util.List;
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ColourClusterVisualTest {
    private static final Path OUT = Paths.get("test_output", "cluster_visual");
    private static final BackgroundId[] BACKGROUNDS = {
        BackgroundId.BG_RANDOM_CIRCLES,
        BackgroundId.BG_RANDOM_LINES,
        BackgroundId.BG_RANDOM_MIXED,
        BackgroundId.BG_SOLID_WHITE,
        BackgroundId.BG_GRADIENT_H_COLOUR,
    };
    @BeforeAll
    void setup() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUT);
    }
    /** Two overlapping circles of the SAME colour must stay in ONE chromatic cluster. */
    @Test
    void sameColourOverlappingCirclesStayInOneCluster() {
        Mat scene = new Mat(480, 640, CvType.CV_8UC3, new Scalar(20, 20, 20));
        Scalar red = new Scalar(0, 0, 200);
        Imgproc.circle(scene, new Point(240, 240), 80, red, -1);
        Imgproc.circle(scene, new Point(340, 240), 80, red, -1);
        List<SceneColourClusters.Cluster> clusters = SceneColourClusters.extract(scene);
        long chromaticCount = clusters.stream().filter(c -> !c.achromatic).count();
        System.out.printf("%nSame-colour overlap: %d chromatic cluster(s) (expect 1)%n", chromaticCount);
        for (int i = 0; i < clusters.size(); i++) {
            var c = clusters.get(i);
            System.out.printf("  cluster[%d]: %s  pixels=%d%n", i,
                c.achromatic ? "achromatic" : String.format("hue=%.0f", c.hue),
                Core.countNonZero(c.mask));
        }
        try {
            int tW = 213, tH = 160;
            Mat panel = new Mat(tH, tW * (1 + clusters.size()), CvType.CV_8UC3, new Scalar(40,40,40));
            Mat thumb = new Mat();
            Imgproc.resize(scene, thumb, new Size(tW, tH));
            thumb.copyTo(panel.submat(0, tH, 0, tW)); thumb.release();
            for (int i = 0; i < clusters.size(); i++) {
                Mat masked = SceneColourClusters.applyMask(scene, clusters.get(i));
                Mat mt = new Mat();
                Imgproc.resize(masked, mt, new Size(tW, tH));
                int x = tW * (i + 1);
                mt.copyTo(panel.submat(0, tH, x, x + tW));
                mt.release(); masked.release();
            }
            Imgcodecs.imwrite(OUT.resolve("same_colour_overlap.png").toString(), panel);
            panel.release();
        } catch (Exception e) { /* visual only */ }
        clusters.forEach(SceneColourClusters.Cluster::release);
        scene.release();
        assert chromaticCount == 1 : "Expected 1 chromatic cluster for two same-colour circles, got " + chromaticCount;
    }
    @Test
    void visualiseClusters() throws IOException {
        for (BackgroundId bgId : BACKGROUNDS) {
            Mat scene = BackgroundFactory.build(bgId, 640, 480);
            List<SceneColourClusters.Cluster> clusters = SceneColourClusters.extract(scene);
            System.out.printf("%n=== %s --- %d clusters ===%n", bgId.name(), clusters.size());
            int tileW = 160, tileH = 120;
            int totalW = tileW * (1 + clusters.size());
            Mat panel = new Mat(tileH, totalW, CvType.CV_8UC3, new Scalar(40, 40, 40));
            Mat thumb = new Mat();
            Imgproc.resize(scene, thumb, new Size(tileW, tileH));
            thumb.copyTo(panel.submat(0, tileH, 0, tileW));
            thumb.release();
            for (int i = 0; i < clusters.size(); i++) {
                SceneColourClusters.Cluster c = clusters.get(i);
                int nonZero = Core.countNonZero(c.mask);
                String label = c.achromatic
                    ? String.format("achromatic %d", nonZero)
                    : String.format("hue=%.0f  %d", c.hue, nonZero);
                System.out.printf("  cluster[%d]: %s%n", i, label);
                Mat masked = SceneColourClusters.applyMask(scene, c);
                Mat maskThumb = new Mat();
                Imgproc.resize(masked, maskThumb, new Size(tileW, tileH));
                int x = tileW * (i + 1);
                maskThumb.copyTo(panel.submat(0, tileH, x, x + tileW));
                Imgproc.putText(panel, label, new Point(x + 2, tileH - 4),
                    Imgproc.FONT_HERSHEY_PLAIN, 0.7, new Scalar(0, 255, 255), 1);
                maskThumb.release();
                masked.release();
            }
            String fname = bgId.name().toLowerCase() + "_clusters.png";
            Imgcodecs.imwrite(OUT.resolve(fname).toString(), panel);
            System.out.printf("  -> %s  (%d clusters)%n", fname, clusters.size());
            clusters.forEach(SceneColourClusters.Cluster::release);
            scene.release();
            panel.release();
        }
    }
}