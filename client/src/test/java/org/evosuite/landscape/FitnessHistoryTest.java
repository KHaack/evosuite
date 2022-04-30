package org.evosuite.landscape;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class FitnessHistoryTest {

    @Test
    public void addTest() {
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 5);
        history.addFitness(2, 0);

        assertEquals(0, history.getObservedMinimum());
        assertEquals(5, history.getObservedMaximum());
    }

    @Test
    public void addInfinityTest() {
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 5);
        history.addFitness(1, Double.POSITIVE_INFINITY);
        history.addFitness(2, 0);
        history.addFitness(3, Double.NEGATIVE_INFINITY);

        assertEquals(0, history.getObservedMinimum());
        assertEquals(5, history.getObservedMaximum());
    }
}