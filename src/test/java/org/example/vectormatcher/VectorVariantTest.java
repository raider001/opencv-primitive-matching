package org.example.vectormatcher;

import org.example.colour.CfMode;
import org.example.MatcherVariant;
import org.example.matchers.VectorVariant;
import org.junit.jupiter.api.*;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Step 2 — VectorVariant enum unit tests.
 *
 * <p>Verifies variant names, epsilon factors, CF modes, and the MatcherVariant
 * contract.  No OpenCV required.
 */
@DisplayName("Vector Step 2 — VectorVariant enum")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class VectorVariantTest {

    @Test @Order(1)
    @DisplayName("2a — Exactly 9 variants are defined")
    void nineVariantsDefined() {
        assertEquals(9, VectorVariant.values().length,
                "Expected exactly 9 VectorVariant entries");
    }

    @Test @Order(2)
    @DisplayName("2b — Each variant has a non-blank name")
    void allVariantNamesNonBlank() {
        for (VectorVariant v : VectorVariant.values()) {
            assertNotNull(v.variantName(), "variantName() must not be null for " + v);
            assertFalse(v.variantName().isBlank(), "variantName() must not be blank for " + v);
        }
    }

    @Test @Order(3)
    @DisplayName("2c — All variant names are unique")
    void allVariantNamesUnique() {
        Set<String> names = MatcherVariant.allNamesOf(VectorVariant.class);
        assertEquals(9, names.size(), "All 9 variant names must be distinct");
    }

    @Test @Order(4)
    @DisplayName("2d — Epsilon factors: STRICT=0.02, NORMAL=0.04, LOOSE=0.08")
    void epsilonFactors() {
        assertEquals(0.02, VectorVariant.VECTOR_STRICT.epsilonFactor(),          1e-9);
        assertEquals(0.02, VectorVariant.VECTOR_STRICT_CF_LOOSE.epsilonFactor(), 1e-9);
        assertEquals(0.02, VectorVariant.VECTOR_STRICT_CF_TIGHT.epsilonFactor(), 1e-9);
        assertEquals(0.04, VectorVariant.VECTOR_NORMAL.epsilonFactor(),          1e-9);
        assertEquals(0.04, VectorVariant.VECTOR_NORMAL_CF_LOOSE.epsilonFactor(), 1e-9);
        assertEquals(0.04, VectorVariant.VECTOR_NORMAL_CF_TIGHT.epsilonFactor(), 1e-9);
        assertEquals(0.08, VectorVariant.VECTOR_LOOSE.epsilonFactor(),           1e-9);
        assertEquals(0.08, VectorVariant.VECTOR_LOOSE_CF_LOOSE.epsilonFactor(),  1e-9);
        assertEquals(0.08, VectorVariant.VECTOR_LOOSE_CF_TIGHT.epsilonFactor(),  1e-9);
    }

    @Test @Order(5)
    @DisplayName("2e — CF modes: base=NONE, CF_LOOSE=LOOSE, CF_TIGHT=TIGHT")
    void cfModes() {
        assertEquals(CfMode.NONE,  VectorVariant.VECTOR_STRICT.cfMode());
        assertEquals(CfMode.LOOSE, VectorVariant.VECTOR_STRICT_CF_LOOSE.cfMode());
        assertEquals(CfMode.TIGHT, VectorVariant.VECTOR_STRICT_CF_TIGHT.cfMode());
        assertEquals(CfMode.NONE,  VectorVariant.VECTOR_NORMAL.cfMode());
        assertEquals(CfMode.LOOSE, VectorVariant.VECTOR_NORMAL_CF_LOOSE.cfMode());
        assertEquals(CfMode.TIGHT, VectorVariant.VECTOR_NORMAL_CF_TIGHT.cfMode());
        assertEquals(CfMode.NONE,  VectorVariant.VECTOR_LOOSE.cfMode());
        assertEquals(CfMode.LOOSE, VectorVariant.VECTOR_LOOSE_CF_LOOSE.cfMode());
        assertEquals(CfMode.TIGHT, VectorVariant.VECTOR_LOOSE_CF_TIGHT.cfMode());
    }

    @Test @Order(6)
    @DisplayName("2f — Variant names contain the level and CF suffix consistently")
    void namingConvention() {
        for (VectorVariant v : VectorVariant.values()) {
            String name = v.variantName();
            assertTrue(name.startsWith("VECTOR_"), "Name must start with VECTOR_: " + name);
            if (v.cfMode() == CfMode.LOOSE) assertTrue(name.endsWith("CF_LOOSE"), name);
            if (v.cfMode() == CfMode.TIGHT) assertTrue(name.endsWith("CF_TIGHT"), name);
            if (v.cfMode() == CfMode.NONE)  assertFalse(name.contains("CF_"), name);
        }
    }

    @Test @Order(7)
    @DisplayName("2g — allNamesOf returns a set of exactly 9 strings")
    void allNamesOfContract() {
        Set<String> names = MatcherVariant.allNamesOf(VectorVariant.class);
        assertEquals(9, names.size());
        assertTrue(names.contains("VECTOR_NORMAL"));
        assertTrue(names.contains("VECTOR_STRICT_CF_TIGHT"));
        assertTrue(names.contains("VECTOR_LOOSE_CF_LOOSE"));
    }
}

