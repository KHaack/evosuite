package org.evosuite.landscape;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NeutralityVolumeTest {

    @Test
    public void simpleTest() {
        NeutralityVolume volume = new NeutralityVolume();
        FitnessHistory history = new FitnessHistory();

        assertEquals(0, volume.neutralityVolume(history));

        history.addFitness(0, 0.3);
        assertEquals(1, volume.neutralityVolume(history));
    }

    @Test
    public void neutralityVolumeTest() {
        NeutralityVolume volume = new NeutralityVolume();
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 0.3);
        history.addFitness(1, 0.3);
        history.addFitness(2, 0.3);

        history.addFitness(3, 0.2);
        history.addFitness(4, 0.2);

        history.addFitness(5, 0.7);
        history.addFitness(6, 0.7);

        assertEquals(3, volume.neutralityVolume(history));
    }


    @Test
    public void repeatTest() {
        NeutralityVolume volume = new NeutralityVolume();
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 0.3);
        history.addFitness(1, 0.3);
        history.addFitness(2, 0.3);

        history.addFitness(3, 0.2);
        history.addFitness(4, 0.2);

        history.addFitness(5, 0.3);
        history.addFitness(6, 0.3);

        assertEquals(3, volume.neutralityVolume(history));
    }

    @Test
    public void selectFitnessSequenceEpsilonTest() {
        NeutralityVolume volume = new NeutralityVolume();
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 10);

        assertEquals(0.1d, volume.selectFitnessSequenceEpsilon(history), 0.0001d);
    }

    @Test
    public void noChangeTest() {
        NeutralityVolume volume = new NeutralityVolume();
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 1.0);

        history.addFitness(1, 0.999);
        history.addFitness(2, 0.999);

        history.addFitness(3, 0.998);
        history.addFitness(4, 0.998);

        history.addFitness(5, 0.997);
        history.addFitness(6, 0.997);

        List<Integer> sequence = volume.fitnessChangesSequence(history);
        assertEquals(4, sequence.size());

        assertEquals(0, sequence.get(0));
        assertEquals(0, sequence.get(1));
        assertEquals(0, sequence.get(2));
        assertEquals(0, sequence.get(3));
    }

    @Test
    public void changeTest() {
        NeutralityVolume volume = new NeutralityVolume();
        FitnessHistory history = new FitnessHistory();

        history.addFitness(0, 1.0);

        history.addFitness(1, 0.9);
        history.addFitness(2, 0.9);

        history.addFitness(3, 0.8);
        history.addFitness(4, 0.8);

        history.addFitness(5, 1.0);
        history.addFitness(6, 1.0);

        history.addFitness(7, 0.999);

        List<Integer> sequence = volume.fitnessChangesSequence(history);
        assertEquals(5, sequence.size());

        assertEquals(0, sequence.get(0));
        assertEquals(-1, sequence.get(1));
        assertEquals(-1, sequence.get(2));
        assertEquals(1, sequence.get(3));
        assertEquals(0, sequence.get(4));
    }
}