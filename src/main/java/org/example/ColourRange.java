package org.example;

import org.opencv.core.Scalar;

/**
 * An HSV colour range used by {@link ColourPreFilter}.
 *
 * <p>OpenCV HSV ranges: H 0–179, S 0–255, V 0–255.
 *
 * @param hsvLower lower bound (inclusive) as a 3-element {@link Scalar} (H, S, V)
 * @param hsvUpper upper bound (inclusive) as a 3-element {@link Scalar} (H, S, V)
 * @param label    human-readable description, e.g. "red_LOOSE" or "cyan_TIGHT"
 */
public record ColourRange(Scalar hsvLower, Scalar hsvUpper, String label) {

    /**
     * True when the hue range wraps around the 0°/180° boundary — i.e. the colour is
     * red or near-red. When {@code wrapsHue} is true, {@link ColourPreFilter} applies
     * two separate {@code Core.inRange} calls and OR-combines the result.
     *
     * <p>A range wraps when {@code hsvLower.val[0] > hsvUpper.val[0]}.
     */
    public boolean wrapsHue() {
        return hsvLower.val[0] > hsvUpper.val[0];
    }

    @Override
    public String toString() {
        return String.format("ColourRange[%s  H %.0f–%.0f  S %.0f–%.0f  V %.0f–%.0f  wrap=%b]",
                label,
                hsvLower.val[0], hsvUpper.val[0],
                hsvLower.val[1], hsvUpper.val[1],
                hsvLower.val[2], hsvUpper.val[2],
                wrapsHue());
    }
}

