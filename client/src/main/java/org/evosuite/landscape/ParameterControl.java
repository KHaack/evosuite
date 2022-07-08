package org.evosuite.landscape;

import org.evosuite.TestGenerationContext;
import org.evosuite.coverage.branch.BranchPool;

import java.util.List;

public class ParameterControl {
    private int generations;
    private int nv;
    private double ic;
    private double fitness;
    private int gradientBranches;
    private int gradientCovered;
    private int branches;
    private boolean branchless;
    private double branchRatio;
    private double neutralityGen;
    private double gradienRatio;

    public ParameterControl(List<Integer> generationList, List<Integer> neutralityVolumes, List<Double> informationContentList, List<Double> fitnessRatio, List<Integer> gradientBranchesCovered, List<Integer> gradientBranchList) {
        generations = generationList.get(generationList.size() - 1);
        nv = neutralityVolumes.get(neutralityVolumes.size() - 1);
        ic = informationContentList.get(informationContentList.size() - 1);
        fitness = fitnessRatio.get(fitnessRatio.size() - 1);
        gradientBranches = gradientBranchList.get(gradientBranchList.size() - 1);
        gradientCovered = gradientBranchesCovered.get(gradientBranchesCovered.size() - 1);
        branches = BranchPool.getInstance(TestGenerationContext.getInstance().getClassLoaderForSUT()).getBranchCounter() * 2;

        branchless = branches == 0;
        branchRatio = branchless ? 0.0 : (double) gradientBranches / (double) branches;
        neutralityGen = generations == 0 ? 0 : (double) nv / generations;
        gradienRatio = gradientBranches == 0 ? 1.0 : (double) gradientCovered / (double) gradientBranches;

    }

    public boolean performsWell() {
        if (branchRatio <= 0.39) {
            if (!branchless) {
                if (fitness <= 0.3) {
                    return neutralityGen <= 0.85;
                } else {
                    return false;
                }
            } else {
                if (neutralityGen <= 0.0) {
                    return false;
                } else {
                    return ic <= 0.39;
                }
            }
        } else {
            if (gradienRatio <= 0.84) {
                if (ic <= 0.69) {
                    return fitness > 0.0;
                } else {
                    return fitness <= 0.02;
                }
            } else {
                if (branchRatio <= 0.59) {
                    return fitness <= 0.93;
                } else {
                    return branchRatio <= 0.67;
                }
            }
        }
    }

    public boolean lowCoverageStd() {
        if(neutralityGen <= 0.0) {
            return branchRatio > 0.24;
        } else {
            return true;
        }
    }

    public boolean relativeHighCoverage() {
        if(neutralityGen <= 0.0) {
            if(branchRatio <= 0.07) {
                return false;
            } else {
                return gradienRatio > 0.76;
            }
        } else {
            if(neutralityGen <= 0.91) {
                return true;
            } else {
                return branchRatio > 0.27;
            }
        }
    }

    public boolean highStDevRelativeLowCoverage() {
        // depth 2, gini, RANDOM 42, pc_at 0.3
        return neutralityGen <= 0.0 && branchRatio <= 0.00;
    }

    public boolean pop125RelativeLowCoverage() {
        if(neutralityGen <= 0) {
            return branchRatio <= 0;
        } else {
            return false;
        }
    }
}
