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
     * The observed minimum fitness value.
     */
    private Double observedMinimum;

    /**
     * Adds the passed fitness value to the history.
     *
     * @param age          The age of the fitness value.
     * @param fitnessValue The fitness value.
     */
    public void addFitness(int age, double fitnessValue) {
        if (null == this.observedMinimum || fitnessValue > this.observedMinimum) {
            this.observedMinimum = fitnessValue;
        }

        if (null == this.observedMaximum || fitnessValue < this.observedMaximum) {
            this.observedMaximum = fitnessValue;
        }

        logger.info("{} -> {}", age, fitnessValue);


        this.fitnessHistory.put(age, fitnessValue);
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

    /**
     * Returns the observed minimum fitness value.
     *
     * @return Returns the observed minimum fitness value.
     */
    public Double getObservedMinimum() {
        return observedMinimum;
    }
}
