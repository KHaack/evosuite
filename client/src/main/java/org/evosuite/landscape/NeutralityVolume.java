package org.evosuite.landscape;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class NeutralityVolume implements Serializable {
    /**
     * The logger.
     */
    private static final Logger logger = LoggerFactory.getLogger(NeutralityVolume.class);
    /**
     * The divider used for the fitness sequence epsilon selection.
     * TODO: via properties
     */
    private static final int SEQUENCE_EPSILON_SELECTION_DIV = 100;
    /**
     * The epsilon for the area compare.
     */
    private static final double NEUTRALITY_VOLUME_EPSILON = 0.000001d;
    /**
     * The underlying fitness history.
     */
    private final FitnessHistory fitnessHistory;
    /**
     * The calculated change sequence.
     */
    private final LinkedList<Integer> changeSequence = new LinkedList<>();
    /**
     * The last change value for the neutrality volume.
     */
    private Double lastChangeValue = null;
    /**
     * The last change type [0, -1, 1] for the change sequence.
     */
    private Integer lastChangeType = null;
    /**
     * The counted areas for the neutrality volume.
     */
    private int areaCount = 0;
    /**
     * The subBlock count mapped by the subBlocks for the information content.
     */
    private HashMap<SubBlock, Integer> subBlocks = new HashMap<>();
    /**
     * The total number of subBlocks.
     */
    private int numberOfSubBlocks = 0;

    /**
     * Constructor.
     *
     * @param fitnessHistory The underlying fitness history.
     */
    public NeutralityVolume(FitnessHistory fitnessHistory) {
        this.fitnessHistory = fitnessHistory;
    }

    /**
     * Initialize the calculations.
     */
    public void init() {
        double sequenceEpsilon = this.selectFitnessSequenceEpsilon();
        for (Map.Entry<Integer, Double> entry : fitnessHistory.getFitnessHistory().entrySet()) {
            this.add(entry.getValue(), sequenceEpsilon);
        }
    }

    /**
     * Returns the fitnessHistory.
     *
     * @return Returns the fitnessHistory.
     */
    public FitnessHistory getFitnessHistory() {
        return fitnessHistory;
    }

    /**
     * Add the passed value to the sequence.
     *
     * @param value           The value.
     * @param sequenceEpsilon The change epsilion.
     */
    private void add(double value, double sequenceEpsilon) {
        if (null == lastChangeValue || Math.abs(lastChangeValue - value) >= NEUTRALITY_VOLUME_EPSILON) {
            this.areaCount++;

            if (null == this.lastChangeValue) {
                this.changeSequence.add(0);
                this.lastChangeType = 0;
            } else {
                double changeValue = this.lastChangeValue - value;

                // get change type
                int changeType;
                if (changeValue > sequenceEpsilon) {
                    changeType = -1;
                } else if (changeValue < -sequenceEpsilon) {
                    changeType = 1;
                } else {
                    changeType = 0;
                }

                // count the sub blocks
                if (this.lastChangeType != null && this.lastChangeType != changeType) {
                    SubBlock block = new SubBlock(this.lastChangeType, changeType);
                    if (!this.subBlocks.containsKey(block)) {
                        this.subBlocks.put(block, 0);
                    }

                    this.subBlocks.put(block, this.subBlocks.get(block) + 1);
                    this.numberOfSubBlocks++;
                }

                this.changeSequence.add(changeType);
                this.lastChangeType = changeType;
            }

            this.lastChangeValue = value;
        }
    }

    /**
     * Select a fitness sequence epsilon based on the fitness history.
     *
     * @return The selected epsilon.
     */
    public double selectFitnessSequenceEpsilon() {
        if (null == this.fitnessHistory.getObservedMaximum()) {
            return 0.0d;
        }

        return this.fitnessHistory.getObservedMaximum() / SEQUENCE_EPSILON_SELECTION_DIV;
    }

    /**
     * Returns the neutrality volume for the {@link FitnessHistory}.
     *
     * @return Returns the neutrality volume.
     */
    public int getNeutralityVolume() {
        return this.areaCount;
    }

    /**
     * Returns the neutrality volume in form of a sequence of fitness changed.
     *
     * @return The fitness changes.
     */
    public List<Integer> getChangeSequence() {
        return this.changeSequence;
    }

    /**
     * Returns the subBlocks.
     *
     * @return Returns the subBlocks.
     */
    public HashMap<SubBlock, Integer> getSubBlocks() {
        return subBlocks;
    }

    /**
     * Returns the information content.
     *
     * @return Returns the information content.
     */
    public double getInformationContent() {
        double informationContent = 0;

        for (Map.Entry<SubBlock, Integer> entry : this.subBlocks.entrySet()) {
            double prop = this.getSubBlockPropability(entry.getKey());
            informationContent += prop * (Math.log(prop) / Math.log(6));
        }

        return -informationContent;
    }

    /**
     * Returns the number of counted subBlocks.
     *
     * @return Returns the number of counted subBlocks.
     */
    public int getNumberOfSubBlocks() {
        return numberOfSubBlocks;
    }

    /**
     * Returns the subBlock propability.
     *
     * @param subBlock The subBlock for that the propability should be calculated.
     * @return Returns the subBlock propability.
     */
    public double getSubBlockPropability(SubBlock subBlock) {
        return this.subBlocks.containsKey(subBlock) ? this.subBlocks.get(subBlock) / (double) this.numberOfSubBlocks : 0;
    }

    /**
     * Prints the history to the log.
     */
    public void printHistory() {
        logger.info("Neutrality Volume (NV): " + getNeutralityVolume());

        List<Integer> changes = this.getChangeSequence();
        logger.info("Fitness min: {}", this.getFitnessHistory().getObservedMinimum());
        logger.info("Fitness max: {}", this.getFitnessHistory().getObservedMaximum());
        logger.info("NV ChangeSequence: {}", changes);
        logger.info("NV ChangeSequence: {}", changes);
        logger.info("Information Content (IC): {}", this.getInformationContent());
        logger.info("----------------------");

        for (Map.Entry<Integer, Double> entry : this.getFitnessHistory().getFitnessHistory().entrySet()) {
            logger.info("{} -> {}", entry.getKey(), entry.getValue());
        }
    }
}
