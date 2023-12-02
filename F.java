import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

public class TransformationEstimation {

    public static boolean main(double[][] A, double[][] B) {
        assert A.length == 3 && A[0].length == 3 && B.length == 3 && B[0].length == 3 : "shape error";

        RealMatrix matrixA = new Array2DRowRealMatrix(A);
        RealMatrix matrixB = new Array2DRowRealMatrix(B);

        double[] lambdaA = new EigenDecomposition(matrixA).getRealEigenvalues();
        double[] lambdaB = new EigenDecomposition(matrixB).getRealEigenvalues();

        System.out.println(Arrays.toString(lambdaA));
        System.out.println(Arrays.toString(lambdaB));

        for (double valueA : lambdaA) {
            if (Arrays.stream(lambdaB).anyMatch(valueB -> Double.compare(valueA, valueB) == 0)) {
                return false;
            }
        }

        RealMatrix eye = new Array2DRowRealMatrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
        RealMatrix s = eye.kroneckerProduct(matrixA).subtract(eye.kroneckerProduct(matrixB.transpose()));

        SingularValueDecomposition svd = new SingularValueDecomposition(s);
        double rank = svd.getRank();
        System.out.println(rank);
        System.out.println(s);

        return true;
    }

    public static double[][] main3(double[][] AA, double[][] BB) {
        int n = AA[0].length / 4;
        int m = AA.length;

        RealMatrix A = new Array2DRowRealMatrix(new double[9 * n][9]);
        RealMatrix b = new Array2DRowRealMatrix(new double[9 * n][1]);

        RealMatrix eye = new Array2DRowRealMatrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});

        for (int i = 0; i < n; i++) {
            RealMatrix Ra = new Array2DRowRealMatrix(Arrays.copyOfRange(AA, 0, 3, 4 * i, 4 * i + 3));
            RealMatrix Rb = new Array2DRowRealMatrix(Arrays.copyOfRange(BB, 0, 3, 4 * i, 4 * i + 3));
            A.setSubMatrix(Ra.kroneckerProduct(eye).subtract(eye.kroneckerProduct(Rb.transpose())).getData(), 9 * i, 0);
        }

        SingularValueDecomposition svd = new SingularValueDecomposition(A);
        RealMatrix x = new Array2DRowRealMatrix(svd.getV().getColumn(svd.getRank() - 1));

        RealMatrix R = new Array2DRowRealMatrix(x.getData()).reshape(3, 3);
        R = R.scalarMultiply(Math.signum(R.getDeterminant()) / Math.pow(Math.abs(R.getDeterminant()), 1.0 / 3.0));

        svd = new SingularValueDecomposition(R);
        R = svd.getU().multiply(svd.getV().transpose());

        if (R.getDeterminant() < 0) {
            RealMatrix diagMatrix = new Array2DRowRealMatrix(new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, -1}});
            R = svd.getU().multiply(diagMatrix).multiply(svd.getV().transpose());
        }

        RealMatrix C = new Array2DRowRealMatrix(new double[3 * n][3]);
        RealMatrix d = new Array2DRowRealMatrix(new double[3 * n][1]);

        for (int i = 0; i < n; i++) {
            RealMatrix Ra = new Array2DRowRealMatrix(Arrays.copyOfRange(AA, 0, 3, 4 * i, 4 * i + 3));
            C.setSubMatrix(eye.subtract(Ra).getData(), 3 * i, 0);
            d.setSubMatrix(Arrays.copyOfRange(AA, 0, 3, 4 * i + 3, 4 * i + 4), 3 * i, 0);
            d.setEntry(3 * i, 0, d.getEntry(3 * i, 0) - R.operate(BB[3 * i + 0][3]));
            d.setEntry(3 * i + 1, 0, d.getEntry(3 * i + 1, 0) - R.operate(BB[3 * i + 1][3]));
            d.setEntry(3 * i + 2, 0, d.getEntry(3 * i + 2, 0) - R.operate(BB[3 * i + 2][3]));
        }

        SingularValueDecomposition svdC = new SingularValueDecomposition(C);
        RealMatrix t = svdC.getSolver().solve(d);

        RealMatrix result = new Array2DRowRealMatrix(4, 4);
        result.setSubMatrix(R.getData(), 0, 0);
        result.setColumnMatrix(3, t);

        return result.getData();
    }

    public static void main(String[] args) {
        double[][] A = {
                {-0.989992, -0.141120, 0, 0, 0.070737, 0, 0.997495, -400},
                {0.141120, -0.989992, 0, 0, 0, 1, 0, 0},
                {0, 0, 1, 0, -0.997495, 0, 0.070737, 400}
        };

        double[][] B = {
                {-0.989992, -0.138307, 0.028036, -26.9559, 0.070737, 0.198172, 0.997612, -309.543},
                {0.138307, -0.911449, 0.387470, -96.1332, -0.198172, 0.963323, -0.180936, 59.0244},
                {-0.028036, 0.387470, 0.921456, 19.4872, -0.977612, -0.180936, 0.107415, 291.177}
        };

        main(A, B);
        double[][] result = main3(A, B);

        for (double[] row : result) {
            System.out.println(Arrays.toString(row));
        }
    }
}
