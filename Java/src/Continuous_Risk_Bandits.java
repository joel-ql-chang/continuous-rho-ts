import java.lang.*;
import java.util.ArrayList;
import java.util.function.Function;
import org.apache.commons.math3.distribution.*;
import org.apache.commons.math3.special.Gamma;

import static java.lang.Math.sqrt;

//CONTINUOUS QUANTITIES

class ProbMeasure {
    Function<Double, Double> _pdf; // initialisation
    AbstractRealDistribution _distribution; // initialisation
    ProbMeasure(AbstractRealDistribution g) {
        _distribution = g;
        _pdf = x -> _distribution.density(x);
    } // instantiation

    Double sample() {
        return _distribution.sample();
    }
}

class ContinuousArm {
    ProbMeasure _probMeasure;
    ContinuousArm(ProbMeasure pm) {
        _probMeasure = pm;
    }
    Double sample() {
        return _probMeasure.sample();
    }
}

class RiskFunctional {
    Function<ProbMeasure, Double> _rho; // initialisation
    RiskFunctional(Function<ProbMeasure, Double> f) { _rho = f; } // instantiation
    Double apply(ProbMeasure pm) {
        return _rho.apply(pm);
    }
}

// DISCRETE QUANTITIES

class DiscreteMeasure {
    double[] _support; // input support
    double[] _probabilityVector; // input probabilityVector
    double[] _probInterval;
    int N; // length of vector

    DiscreteMeasure(double[] s, double[] p) {
        _support = s;
        _probabilityVector = p;
        N = p.length;
        _probInterval = new double[N+1];
        for (int i = 0; i < N; i++) {
            _probInterval[i+1] = _probInterval[i] + _probabilityVector[i];
        }
    }

    int sample() {
        double r = Math.random();
        for (int i = 0; i < N; i++) {
            if (r >= _probInterval[i] && r < _probInterval[i+1]) {
                return i;
            }
        }
        return N;
    }
}

class DiscreteArm {
    DiscreteMeasure _discreteMeasure;
    DiscreteArm(DiscreteMeasure dm) {
        _discreteMeasure = dm;
    }
    int sample() {
        return _discreteMeasure.sample();
    }
}

class DiscreteRiskFunctional {
    Function<DiscreteMeasure, Double> _rho; // initialisation
    DiscreteRiskFunctional(Function<DiscreteMeasure, Double> f) { _rho = f; } // instantiation
    Double apply(DiscreteMeasure dm) {
        return _rho.apply(dm);
    }
}

// RHO MTS

class RhoMTS {
    DiscreteRiskFunctional _rho;

    int _horizon;
    int _numIntervals;
    int _sizeIntervals;
    int _numBandits;
    int _supportSize;
    DiscreteArm[] _armDist;
    double[] _support;
    int[][] _priors;
    int[][] _pullCount;

    RhoMTS(DiscreteRiskFunctional r, int n, int L, double[] s, DiscreteArm[] a) {
        _rho = r;
        _horizon = n;
        _numIntervals = L;
        _sizeIntervals = n/L;
        _support = s;
        _armDist = a;
        _numBandits = _armDist.length;
        _supportSize = _support.length;
        _priors = new int[_numBandits][_supportSize];
        for (int i = 0; i < _numBandits; i++) {
            for (int j = 0; j < _supportSize; j++) {
                _priors[i][j] = 1;
            }
        }
        _pullCount = new int[_numBandits][_numIntervals];
    }

    int[][] runAlgorithm() {
        int action;
        int reward;
        double[] samples = new double[_supportSize];
        double[] testRho = new double[_numBandits];

        for (int t = 0; t < _horizon; t++) {
            action = 0;
            if (t < _numBandits) {
                action = t;
            } else {
                for (int k = 0; k < _numBandits; k++) {
                    int[] alphas = _priors[k];

                    double[] gammaSamples = new double[alphas.length];
                    double sum = 0;
                    for (int i = 0; i < alphas.length; i++) {
                        GammaDistribution gamma = new GammaDistribution(alphas[i], 1.0);
                        gammaSamples[i] = gamma.sample();
                        sum += gammaSamples[i];
                    }

                    for (int i = 0; i < alphas.length; i++) {
                        samples[i] = gammaSamples[i]/sum;
                    }
                    DiscreteMeasure sampleDist = new DiscreteMeasure(_support, samples);
                    testRho[k] = _rho.apply(sampleDist);
                }
                for (int k = 0; k < _numBandits; k++) {
                    if (testRho[k] > testRho[action]) {
                        action = k;
                    }
                }
            }

            reward = _armDist[action].sample();
            _pullCount[action][t / _sizeIntervals]++;
            _priors[action][reward]++;
        }
        for (int k = 0; k < _numBandits; k++) {
            for (int l = 0; l < _numIntervals-1; l++) {
                _pullCount[k][l+1] = _pullCount[k][l] + _pullCount[k][l+1];
            }
        }
        return _pullCount;
    }
}

// RHO NPTS

class RhoNPTS {
    RiskFunctional _rho;
    DiscreteRiskFunctional _discreteRho;

    int _horizon;
    int _numIntervals;
    int _sizeIntervals;
    int _numBandits;
    ContinuousArm[] _armDist;
    int[] _priors;
    ArrayList<ArrayList<Double>> _armData;
    int[][] _pullCount;

    RhoNPTS(RiskFunctional r, DiscreteRiskFunctional d, int n, int L, ContinuousArm[] a) {
        _rho = r;
        _discreteRho = d;
        _horizon = n;
        _numIntervals = L;
        _sizeIntervals = n/L;
        _armDist = a;
        _numBandits = _armDist.length;
        _priors = new int[_numBandits];
        _armData = new ArrayList<>(_numBandits);
        for (int i = 0; i < _numBandits; i++) {
            _priors[i] = 1;
            ArrayList<Double> a1 = new ArrayList<Double>();
            a1.add((double) 1);
            _armData.add(a1);
        }
        _pullCount = new int[_numBandits][_numIntervals];
    }

    int[][] runAlgorithm() {
        int action;
        double reward;
        double[] testRho = new double[_numBandits];
        double[] samples;

        for (int t = 0; t < _horizon; t++) {
            action = 0;
            for (int k = 0; k < _numBandits; k++) {
                int alphas = _priors[k];

                // Generating L_k^t
                samples = new double[alphas];
                double[] gammaSamples = new double[alphas];
                double sum = 0;
                for (int i = 0; i < alphas; i++) {
                    GammaDistribution gamma = new GammaDistribution(1.0, 1.0);
                    gammaSamples[i] = gamma.sample();
                    sum += gammaSamples[i];
                }
                for (int i = 0; i < alphas; i++) {
                    samples[i] = gammaSamples[i]/sum;
                }

                // Computing \rho(L_K^t)
                double[] dblArray = new double[alphas];
                for (int i = 0; i < alphas; i++) {
                    dblArray[i] = _armData.get(k).get(i);
                }
                DiscreteMeasure sampleDist = new DiscreteMeasure(dblArray, samples);
                testRho[k] = _discreteRho.apply(sampleDist);
            }
            for (int k = 0; k < _numBandits; k++) {
                if (testRho[k] > testRho[action]) {
                    action = k;
                }
            }
            reward = _armDist[action].sample();
            _armData.get(action).add(reward);
            _pullCount[action][t / _sizeIntervals]++;
            _priors[action]++;
        }
        for (int k = 0; k < _numBandits; k++) {
            for (int l = 0; l < _numIntervals-1; l++) {
                _pullCount[k][l+1] = _pullCount[k][l] + _pullCount[k][l+1];
            }
        }
        return _pullCount;
    }
}

public class Continuous_Risk_Bandits {

    public static Double KLDivergence(BetaDistribution F, BetaDistribution G) {
        Double alphaF = F.getAlpha();
        Double betaF = F.getBeta();
        Double alphaG = G.getAlpha();
        Double betaG = G.getBeta();

        Double term1 = Gamma.gamma(alphaF + betaF) * Gamma.gamma(alphaG) * Gamma.gamma(betaG);
        Double term2 = Gamma.gamma(alphaG + betaG) * Gamma.gamma(alphaF) * Gamma.gamma(betaF);
        Double term3 = (alphaF - alphaG) * (Gamma.digamma(alphaF) - Gamma.digamma(alphaF + betaF));
        Double term4 = (betaF - betaG) * (Gamma.digamma(betaF) - Gamma.digamma(alphaF + betaF));

        return Math.log(term1 / term2) + term3 + term4;
    }

    //Discrete Expectation Functional
    public static Double averaging(double[] s, double[] p) {
        int N = s.length;
        double sum = 0;
        for (int i = 0; i < N; i++) {
            sum += s[i] * p[i];
        }
        return sum;
    }

    //Continuous Expectation Functional
    public static Double integrate(double a, double b, int N, Function<Double,Double> f) {
        double h = (b - a) / N;              // step size
        double sum = 0.5 * (f.apply(a) + f.apply(b));    // area
        for (int i = 1; i < N; i++) {
            double x = a + h * i;
            sum = sum + f.apply(x);
        }
        return sum * h;
    }

    public static Double expectation(ProbMeasure pm) {
        return pm._distribution.getNumericalMean();
    }
    public static Double expectation(double[] s, double[] p) {
        int N = s.length;
        double sum = 0;
        for (int i = 0; i < N; i++) {
            sum += s[i] * p[i];
        }
        return sum;
    }

    public static Double variance(ProbMeasure pm) {
        return pm._distribution.getNumericalVariance();
    }
    public static Double variance(double[] s, double[] p) {
        int N = s.length;
        double sumSquares = 0;
        for (int i = 0; i < N; i++) {
            sumSquares += (s[i] * s[i]) * p[i];
        }
        return sumSquares - expectation(s, p) * expectation(s, p);
    }

    public static Double meanVariance(double gamma, ProbMeasure pm) {
        return gamma*expectation(pm) - variance(pm);
    }
    public static Double meanVariance(double gamma, double[] s, double[] p) {
        return gamma*expectation(s,p) - variance(s,p);
    }

    public static Double DRF(double a, double b, int N, Function<Double,Double> g, ProbMeasure pm) {
        return integrate(a, b, N, x -> g.apply(1 - pm._distribution.probability(a, x)));
    }
    public static Double DRF(Function<Double,Double> g, double[] s, double[] p) {
        int M = s.length-1;
        double sum = 0;
        double[] tailProbability = new double[M+1];
        tailProbability[M] = p[M];
        for (int i = 1; i < M+1; i++) {
            tailProbability[M-i] = tailProbability[M+1-i] + p[M-i];
        }
        for (int i = 0; i < M; i++) {
            sum += (s[i+1] - s[i]) * g.apply(tailProbability[i+1]);
        }
        return sum;
    }

    public static Double CVaRDistortionFunction(double x, double a) {
        if (x < 1 - a) {
            return x/(1-a);
        }
        return (double) 1;
    }

    public static Double PropDistortionFunction(double x, double p) {
        return Math.pow(x, p);
    }

    public static Double LookbackDistortionFunction(double x, double p) {
        return Math.pow(x, p) * (1 - p * Math.log(x));
    }

    public static Double CVaR(double a, double b, int N, double alpha, ProbMeasure pm) {
        return DRF(a, b, N, x -> CVaRDistortionFunction(x, alpha), pm);
    }
    public static Double CVaR(double alpha, double[] s, double[] p) {
        return DRF(x -> CVaRDistortionFunction(x, alpha), s, p);
    }

    public static Double Prop(double a, double b, int N, double p, ProbMeasure pm) {
        return DRF(a, b, N, x -> PropDistortionFunction(x, p), pm);
    }
    public static Double Prop(double prop, double[] s, double[] p) {
        return DRF(x -> PropDistortionFunction(x, prop), s, p);
    }

    public static Double Lookback(double a, double b, int N, double p, ProbMeasure pm) {
        return DRF(a, b, N, x -> LookbackDistortionFunction(x, p), pm);
    }
    public static Double Lookback(double prop, double[] s, double[] p) {
        return DRF(x -> LookbackDistortionFunction(x, prop), s, p);
    }

    public static Double rhoOne(double gamma, double alpha, ProbMeasure pm) {
        return meanVariance(gamma, pm) + CVaR(0, 1, 100, alpha, pm);
    }
    public static Double rhoOne(double gamma, double alpha, double[] s, double[] p) {
        return meanVariance(gamma, s, p) + CVaR(alpha, s, p);
    }

    public static Double rhoTwo(double prop1, double prop2, ProbMeasure pm) {
        int numDivisions = 10000;
        double slack = 0.0001;
        return Prop(0, 1, numDivisions, prop1, pm) + Lookback(slack, 1-slack, numDivisions, prop2, pm);
    }
    public static Double rhoTwo(double prop1, double prop2, double[] s, double[] p) {
        return Prop(prop1, s, p) + Lookback(prop2, s, p);
    }

    public static void main(String[] args) {
        boolean finiteAlphabet = false;
        int n = 5000; // horizon
        int L = 10; // numIntervals
        int M = 4; // supportSize
        int K = 3; // numBandits
        int numExperiments = 50; // numExperiments
        boolean comments = true; // On comments
        boolean armPulls = false; // On comments for arm pulls per experiment
        boolean rhoOne = false; // rhoOne or rhoTwo

        double[] rhoValues = new double[K];
        double[] regretGaps = new double[K];
        double[][] totalRegret = new double[n][L];
        int[][] tempPullCount; //Collect Regret

        double MVparameter = 0.5;
        double CVaRparameter = 0.95;
        double Propparameter = 0.7;
        double Lookbackparameter = 0.6;

        if (finiteAlphabet) {
            DiscreteRiskFunctional myRiskFunctional;
            DiscreteArm[] arms;
            RhoMTS algorithm;
            myRiskFunctional = new DiscreteRiskFunctional(x -> averaging(x._support, x._probabilityVector));
            // choice of rho

            double[] s = new double[M+1];
            for (int i = 0; i < M+1; i++) {
                s[i] = (double) (i/M);
            }

            arms = new DiscreteArm[K];

            for (int N = 0; N < numExperiments; N++) {
                //Set up of new bandits
                for (int i = 0; i < K; i++) {
                    double sum = 0;
                    double[] gammaSamples = new double[s.length];
                    double[] samples = new double[s.length];
                    for (int j = 0; j < s.length; j++) {
                        GammaDistribution gamma = new GammaDistribution(1, 1.0);
                        gammaSamples[j] = gamma.sample();
                        sum += gammaSamples[j];
                    }
                    for (int j = 0; j < s.length; j++) {
                        samples[j] = gammaSamples[j]/sum;
                    }
                    arms[i] = new DiscreteArm(new DiscreteMeasure(s, samples));
                    rhoValues[i] = myRiskFunctional.apply(arms[i]._discreteMeasure);
                }

                //Optimal arm and regret data
                int optimal = 0;
                for (int i = 0; i < K; i++) {
                    if (rhoValues[i] > rhoValues[optimal]) {
                        optimal = i;
                    }
                }
                for (int i = 0; i < K; i++) {
                    regretGaps[i] = rhoValues[optimal] - rhoValues[i];
                }

                //Run experiment
                algorithm = new RhoMTS(myRiskFunctional, n, L, s, arms);
                tempPullCount = algorithm.runAlgorithm();
                for (int l = 0; l < L; l++) {
                    for (int k = 0; k < K; k++) {
                        totalRegret[N][l] += tempPullCount[k][l] * regretGaps[k];
                    }
                }
            }
        } else {
            RiskFunctional myRiskFunctional;
            DiscreteRiskFunctional myDiscreteRiskFunctional;
            ContinuousArm[] arms;
            RhoNPTS algorithm;

            if (rhoOne) {
                myRiskFunctional = new RiskFunctional(x -> rhoOne(MVparameter, CVaRparameter, x));
                myDiscreteRiskFunctional = new DiscreteRiskFunctional(x -> rhoOne(MVparameter, CVaRparameter, x._support, x._probabilityVector));
            } else {
                myRiskFunctional = new RiskFunctional(x -> rhoTwo(Propparameter, Lookbackparameter, x));
                myDiscreteRiskFunctional = new DiscreteRiskFunctional(x -> rhoTwo(Propparameter, Lookbackparameter, x._support, x._probabilityVector));
            }
            // choice of rho

            arms = new ContinuousArm[K];

            //Set up of new bandits

            int[][] betaDetails = {{1,3}, {3,3}, {3,1}};
            int[] testKInfBetaDetails;
            if (rhoOne) {
                testKInfBetaDetails = new int[]{7, 2};
            } else {
                testKInfBetaDetails = new int[]{8, 2};
            }

            if (comments) {
                System.out.println("Rho values: ");
            }
            for (int i = 0; i < K; i++) {
                arms[i] = new ContinuousArm(new ProbMeasure(new BetaDistribution(betaDetails[i][0], betaDetails[i][1])));
                rhoValues[i] = myRiskFunctional.apply(arms[i]._probMeasure);

                if (comments) {
                    System.out.println("Arm " + i + ", rho: " + rhoValues[i]);
                }
            }

            //Optimal arm and regret data
            int optimal = 0;
            for (int i = 0; i < K; i++) {
                if (rhoValues[i] > rhoValues[optimal]) {
                    optimal = i;
                }
            }
            for (int i = 0; i < K; i++) {
                regretGaps[i] = rhoValues[optimal] - rhoValues[i];
            }
            if (comments) {
                System.out.println("Optimal: " + optimal);
            }

            ContinuousArm testKInf = new ContinuousArm(new ProbMeasure(new BetaDistribution(testKInfBetaDetails[0], testKInfBetaDetails[1])));
            if (comments) {
                System.out.println("Test KInf Arm, rho: " + myRiskFunctional.apply(testKInf._probMeasure));
                double lowerBoundConstant = 0;
                double computedKL;
                double finalLowerBoundConstant = 0;
                for (int i = 0; i < K; i++) {
                    computedKL = KLDivergence(new BetaDistribution(testKInfBetaDetails[0], testKInfBetaDetails[1]), new BetaDistribution(betaDetails[i][0], betaDetails[i][1]));
                    lowerBoundConstant += 1/computedKL;
                    finalLowerBoundConstant += regretGaps[i]/computedKL;
                    System.out.println("KL divg: " + computedKL);
                }
                System.out.println("Lower bound constant: " + lowerBoundConstant);

                System.out.print("Lower bound constant intervals: ");
                System.out.print(finalLowerBoundConstant * Math.log(n/(2*L)) + ",");
                for (int l = 1; l < L; l++) {
                    System.out.print(finalLowerBoundConstant * Math.log((l*n/L)) + ",");
                }
                System.out.print(finalLowerBoundConstant * Math.log((n)) + ",");
                System.out.println("");
            }

            for (int N = 0; N < numExperiments; N++) {
                //Run experiment
                algorithm = new RhoNPTS(myRiskFunctional, myDiscreteRiskFunctional, n, L, arms);
                tempPullCount = algorithm.runAlgorithm();
                if (armPulls) {
                    System.out.print("Arm Pulls: ");
                    for (int k = 0; k < K; k++) {
                        System.out.print(tempPullCount[k][L-1] + ",");
                    }
                    System.out.println();
                }
                for (int l = 0; l < L; l++) {
                    for (int k = 0; k < K; k++) {
                        totalRegret[N][l] += tempPullCount[k][l] * regretGaps[k];
                    }
                }
                if ((N+1) % 5 == 0) {
                    System.out.println("Experiment " + N);
                }
            }
        }
        //Data collection
        double[] mean = new double[L];
        double[] stdDev = new double[L];
        for (int l = 0; l < L; l++) {
            double sum = 0;
            double sumSquares = 0;
            for (int N = 0; N < numExperiments; N++) {
                sum += totalRegret[N][l];
                sumSquares += (totalRegret[N][l] * totalRegret[N][l]);
            }
            mean[l] = sum / numExperiments;
            stdDev[l] = sqrt(sumSquares / numExperiments - mean[l] * mean[l]);
        }
        System.out.print("Mean Regret: ");
        for (int l = 0; l < L; l++) {
            System.out.print(mean[l] + "," );
        }
        System.out.println("");
        System.out.print("Mean Std Dev Regret: ");
        for (int l = 0; l < L; l++) {
            System.out.print(stdDev[l] + "," );
        }
    }
}
