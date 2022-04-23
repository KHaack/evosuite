package org.evosuite.landscape;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;

import static org.junit.Assert.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class NeutralityVolumeTest {

    @Test
    public void simpleTest() {
        FitnessHistory history = new FitnessHistory();

        NeutralityVolume volume = new NeutralityVolume(history);
        volume.init();
        assertEquals(0, volume.getNeutralityVolume());

        volume = new NeutralityVolume(history);
        history.addFitness(0, 0.3);
        volume.init();

        assertEquals(1, volume.getNeutralityVolume());
    }

    @Test
    public void neutralityVolumeTest() {
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 0.3);
        history.addFitness(1, 0.3);
        history.addFitness(2, 0.3);

        history.addFitness(3, 0.2);
        history.addFitness(4, 0.2);

        history.addFitness(5, 0.7);
        history.addFitness(6, 0.7);

        NeutralityVolume volume = new NeutralityVolume(history);
        volume.init();
        assertEquals(3, volume.getNeutralityVolume());
    }


    @Test
    public void repeatTest() {
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 0.3);
        history.addFitness(1, 0.3);
        history.addFitness(2, 0.3);

        history.addFitness(3, 0.2);
        history.addFitness(4, 0.2);

        history.addFitness(5, 0.3);
        history.addFitness(6, 0.3);

        NeutralityVolume volume = new NeutralityVolume(history);
        volume.init();
        assertEquals(3, volume.getNeutralityVolume());
    }

    @Test
    public void selectFitnessSequenceEpsilonTest() {
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 10);

        NeutralityVolume volume = new NeutralityVolume(history);
        volume.init();
        assertEquals(0.1d, volume.selectFitnessSequenceEpsilon(), 0.0001d);
    }

    @Test
    public void noChangeTest() {
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 1.0);

        history.addFitness(1, 0.999);
        history.addFitness(2, 0.999);

        history.addFitness(3, 0.998);
        history.addFitness(4, 0.998);

        history.addFitness(5, 0.997);
        history.addFitness(6, 0.997);

        NeutralityVolume volume = new NeutralityVolume(history);
        volume.init();
        List<Integer> sequence = volume.getChangeSequence();
        assertEquals(4, sequence.size());

        assertEquals(0, sequence.get(0));
        assertEquals(0, sequence.get(1));
        assertEquals(0, sequence.get(2));
        assertEquals(0, sequence.get(3));
    }

    @Test
    public void changeTest() {
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 1.0);

        history.addFitness(1, 0.9);
        history.addFitness(2, 0.9);

        history.addFitness(3, 0.8);
        history.addFitness(4, 0.8);

        history.addFitness(5, 1.0);
        history.addFitness(6, 1.0);

        history.addFitness(7, 0.999);

        NeutralityVolume volume = new NeutralityVolume(history);
        volume.init();
        List<Integer> sequence = volume.getChangeSequence();
        assertEquals(5, sequence.size());

        assertEquals(0, sequence.get(0));
        assertEquals(-1, sequence.get(1));
        assertEquals(-1, sequence.get(2));
        assertEquals(1, sequence.get(3));
        assertEquals(0, sequence.get(4));
    }

    @Test
    public void subBlockTest() {
        FitnessHistory history = new FitnessHistory();

        // 0
        // -1
        history.addFitness(0, 1.0);
        history.addFitness(1, 0.9);
        history.addFitness(2, 0.9);

        // -1
        history.addFitness(3, 0.8);
        history.addFitness(4, 0.8);

        // 1
        history.addFitness(5, 1.0);
        history.addFitness(6, 1.0);

        // 0
        history.addFitness(7, 0.999);

        NeutralityVolume volume = new NeutralityVolume(history);
        volume.init();

        HashMap<SubBlock, Integer> subBlocks = volume.getSubBlocks();
        assertEquals(3, subBlocks.size());

        assertTrue(subBlocks.containsKey(new SubBlock(0, -1)));
        assertTrue(subBlocks.containsKey(new SubBlock(-1, 1)));
        assertTrue(subBlocks.containsKey(new SubBlock(1, 0)));

        assertEquals(1, subBlocks.get(new SubBlock(0, -1)));
        assertEquals(1, subBlocks.get(new SubBlock(-1, 1)));
        assertEquals(1, subBlocks.get(new SubBlock(1, 0)));
    }

    @Test
    public void informationContentTest() {
        FitnessHistory history = new FitnessHistory();

        // 0
        // -1
        history.addFitness(0, 1.0);
        history.addFitness(1, 0.9);
        history.addFitness(2, 0.9);

        // -1
        history.addFitness(3, 0.8);
        history.addFitness(4, 0.8);

        // 1
        history.addFitness(5, 1.0);
        history.addFitness(6, 1.0);

        // 0
        history.addFitness(7, 0.999);

        NeutralityVolume volume = new NeutralityVolume(history);
        volume.init();

        HashMap<SubBlock, Integer> subBlocks = volume.getSubBlocks();
        assertEquals(1, subBlocks.get(new SubBlock(0, -1)));
        assertEquals(1, subBlocks.get(new SubBlock(-1, 1)));
        assertEquals(1, subBlocks.get(new SubBlock(1, 0)));

        assertEquals(3, volume.getNumberOfSubBlocks());

        assertEquals(0.33333d, volume.getSubBlockPropability(new SubBlock(0, -1)), 0.0001d);
        assertEquals(0.33333d, volume.getSubBlockPropability(new SubBlock(-1, 1)), 0.0001d);
        assertEquals(0.33333d, volume.getSubBlockPropability(new SubBlock(1, 0)), 0.0001d);

        assertEquals(0.6131d, volume.getInformationContent(), 0.0001d);
    }
}