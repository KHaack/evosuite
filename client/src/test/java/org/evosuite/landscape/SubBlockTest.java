package org.evosuite.landscape;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SubBlockTest {

    @Test
    public void testHashCode() {
        assertNotEquals(new SubBlock(0, 1), new SubBlock(1, 0));
        assertNotEquals(new SubBlock(0, 1), new SubBlock(-1, 0));
        assertNotEquals(new SubBlock(0, 1), new SubBlock(0, -1));


        assertNotEquals(new SubBlock(1, -1), new SubBlock(1, 0));
        assertNotEquals(new SubBlock(1, -1), new SubBlock(-1, 0));
        assertNotEquals(new SubBlock(1, -1), new SubBlock(0, -1));
    }
}