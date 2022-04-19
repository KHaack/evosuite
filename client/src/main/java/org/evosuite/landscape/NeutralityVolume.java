package org.evosuite.landscape;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Based on the work of Nasser M. Albunian. This class measure the neutrality based on the number of neighbouring
 * fitness values with equal value.
 *
 * @author Kevin Haack
 */
public class NeutralityVolume {
    /**
     * The epsilon for the area compare.
     */
    private static final double NEUTRALITY_VOLUME_EPSILON = 0.000001d;
    /**
     * The divider used for the fitness sequence epsilon selection.
     */
    private static final int SEQUENCE_EPSILON_SELECTION_DIV = 100;

    /**
     * The constructor.
     */
    public NeutralityVolume() {
    }

    /**
     * Returns the neutrality volume for the passed {@link FitnessHistory}.
     *
     * @param fitnessHistory The fitness history.
     * @return Returns the neutrality volume.
     */
    public int neutralityVolume(FitnessHistory fitnessHistory) {
        int areas = 0;

        Double lastValue = null;
        for (Map.Entry<Integer, Double> entry : fitnessHistory.getFitnessHistory().entrySet()) {
            if (null == lastValue || Math.abs(lastValue - entry.getValue()) >= NEUTRALITY_VOLUME_EPSILON) {
                lastValue = entry.getValue();
                areas++;
            }
        }

        return areas;
    }

    /**
     * Returns the neutrality volume in form of a sequence of fitness changed.
     *
     * @param fitnessHistory The fitness history.
     * @return The fitness changes.
     */
    public List<Integer> fitnessChangesSequence(FitnessHistory fitnessHistory) {
        LinkedList<Integer> changeSequence = new LinkedList<>();

        Double lastValue = null;
        double sequenceEpsilon = selectFitnessSequenceEpsilon(fitnessHistory);

        for (Map.Entry<Integer, Double> entry : fitnessHistory.getFitnessHistory().entrySet()) {
            if (null == lastValue || Math.abs(lastValue - entry.getValue()) >= NEUTRALITY_VOLUME_EPSILON) {
                if (null == lastValue) {
                    changeSequence.add(0);
                } else {
                    double change = lastValue - entry.getValue();
                    if (change > sequenceEpsilon) {
                        changeSequence.add(-1);
                    } else if (change < -sequenceEpsilon) {
                        changeSequence.add(1);
                    } else {
                        changeSequence.add(0);
                    }
                }

                lastValue = entry.getValue();
            }
        }

        return changeSequence;
    }

    /**
     * Select a fitness sequence epsilon based on the passed {@link FitnessHistory}.
     *
     * @param fitnessHistory The fitness history.
     * @return The selected epsilon.
     */
    public double selectFitnessSequenceEpsilon(FitnessHistory fitnessHistory) {
        return fitnessHistory.getObservedMaximum() / SEQUENCE_EPSILON_SELECTION_DIV;
    }
}
