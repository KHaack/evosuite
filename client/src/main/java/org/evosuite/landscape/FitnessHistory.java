package org.evosuite.landscape;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents the fitness history of the search.
 *
 * @author Kevin Haack
 */
public class FitnessHistory implements Serializable {
    /**
     * The logger.
     */
    private static final Logger logger = LoggerFactory.getLogger(FitnessHistory.class);
    /**
     * The fitness history, mapped by the age.
     */
    private final LinkedHashMap<Integer, Double> fitnessHistory = new LinkedHashMap<>();
    /**
     * The observed maximum fitness value.
     */
    private Double observedMaximum;

    /**
     * Adds the passed fitness value to the history.
     *
     * @param age          The age of the fitness value.
     * @param fitnessValue The fitness value.
     */
    public void addFitness(int age, double fitnessValue) {
        if (null == this.observedMaximum || fitnessValue > this.observedMaximum) {
            this.observedMaximum = fitnessValue;
        }

        this.fitnessHistory.put(age, fitnessValue);
    }

    /**
     * Prints the fitness history.
     */
    public void printHistory() {
        NeutralityVolume nv = new NeutralityVolume(this);
        nv.init();
        logger.info("Neutrality Volume (NV): " + nv.getNeutralityVolume());

        List<Integer> changes = nv.getChangeSequence();
        logger.info("NV ChangeSequence: {}", changes);
        logger.info("Information Content (IC): {}", nv.getInformationContent());
        logger.info("----------------------");

        for (Map.Entry<Integer, Double> entry : this.fitnessHistory.entrySet()) {
            logger.info("{} -> {}", entry.getKey(), entry.getValue());
        }
    }

    /**
     * Returns the fitness history.
     *
     * @return Returns the fitness history.
     */
    public LinkedHashMap<Integer, Double> getFitnessHistory() {
        return fitnessHistory;
    }

    /**
     * Returns the observed maximum fitness value.
     *
     * @return Returns the observed maximum fitness value.
     */
    public Double getObservedMaximum() {
        return observedMaximum;
    }
}
