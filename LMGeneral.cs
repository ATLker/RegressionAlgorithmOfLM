using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;

using System;
using System.Diagnostics;
using System.Threading.Tasks;
namespace LM
{
    class LMGeneral
    {
        public static Vector<double> Regression(
       double[] data,      //数据点的X
       double[] obs,       //数据点的Y
       double[] parameters,//初始参数设定值，迭代的就是这些参数
       Func<Func<double[]>, double, double> function,//要拟合的函数,第一个泛型返回数组参数，第二个代表x，第三代表y
       double lamda = 0.5,//初始lamda设定值
       int iterations = 50,//最大迭代次数
       double increase = 4,//lamda增大量级
       double decrease = 3)//lamda减小量级
        {
            if (data.Length != obs.Length)
            {
                throw new Exception("Observation Data and Correspaned Data are not the same dimention!");
            }
            var dataLength = data.Length;            //将数组长度存为常量
            var numberOfParams = parameters.Length;            //将数组长度存为常量
            double[] parametersLm = new double[numberOfParams];     //迭代一次后的参数
            var zhuangtai = 1;                                      //减少计算量的状态判断数，在不更新参数时改为0减少下一次迭代计算量
            Vector<double> ev = CreateVector.Dense<double>(dataLength);    //误差向量
            Vector<double> evLm = CreateVector.Dense<double>(dataLength);   //参数迭代一次后的误差向量
            double error = 0;                                               //数据总误差（距离平方之和）
            Matrix<double> jacobMatrix = CreateMatrix.Dense<double>(dataLength, numberOfParams);//误差函数的雅可比矩阵
            Matrix<double> hessianMatrix = CreateMatrix.Dense<double>(numberOfParams, numberOfParams);//误差函数的黑塞矩阵
            //循环开始
            for (int i = 0; i < iterations; i++)
            {
                //Console.WriteLine($"第{i}次迭代开始");
                if (zhuangtai == 1)
                {
                    //计算误差向量
                    for (int j = 0; j < dataLength; j++)
                    {
                        ev[j] = obs[j] - function(() => parameters, data[j]);
                    }
                    //计算误差.
                    error = ev.DotProduct(ev);
                    //计算雅可比矩阵
                    jacobMatrix = JacobMatrix(function, parameters, data);
                    //jacobMatrix = JacobMatrix_special(parameters, data);
                    //黑塞矩阵
                    hessianMatrix = jacobMatrix.TransposeThisAndMultiply(jacobMatrix);
                }
                //生成单位矩阵
                var I = CreateMatrix.DenseDiagonal(numberOfParams, numberOfParams, lamda);
                //H=J*J+lamda*I.  算法的灵魂所在!!!
                var hessianMatrixPlus = hessianMatrix + lamda * I;
                // H矩阵过小的话说明步长会很小.
                if (hessianMatrixPlus.Determinant() < 1.0e-10)
                {
                    //Console.WriteLine("H矩阵过小.");
                    break;
                }
                // 计算更新步长dp.
                var dp = hessianMatrixPlus.Inverse()
                    .Multiply(jacobMatrix.TransposeThisAndMultiply(ev));
                //先暂时更新参数存到中间变量里
                for (int j = 0; j < numberOfParams; j++)
                {
                    parametersLm[j] = parameters[j] + dp[j];
                }

                // 计算新参数的误差.
                for (int j = 0; j < dataLength; j++)
                {
                    evLm[j] = obs[j] - function(() => parametersLm, data[j]);
                }
                var errorLm = evLm.DotProduct(evLm);
                // 比一下和之前没更新时误差谁大
                if (errorLm > error)
                {
                    //若误差增大，跳过本次更新，增大lamda，继续下一次寻找
                    lamda *= increase;
                    zhuangtai = 0;
                    continue;
                }
                //误差减小的话缩小Lamda,更新参数
                lamda /= decrease;
                zhuangtai = 1;
                for (int j = 0; j < numberOfParams; j++)
                {
                    parameters[j] = parametersLm[j];
                }
                //如果更新步长过小就结束
                if (dp.AbsoluteMaximum() < 1.0e-3)
                {
                    //Console.WriteLine($"步长过小,总用时{totalltime}毫秒");
                    break;
                }
            }
            var parametersVector = CreateVector.DenseOfArray<double>(parameters);
            return parametersVector;
        }
        private static Matrix<double> JacobMatrix(Func<Func<double[]>, double, double> function, double[] parameters, double[] data)
        {
            int length = data.Length;
            int numberofparameters = parameters.Length;
            Matrix<double> jacobMatrix = CreateMatrix.Dense<double>(length, numberofparameters);
            for (int i = 0; i < length; i++)
            {
                for (int j = 0; j < numberofparameters; j++)
                {
                    jacobMatrix[i, j] = Differentiate.Derivative(x =>
                             function(() => {
                                 var newParameters = parameters;
                                 newParameters[j] = x;
                                 return newParameters;
                             }, data[i]), parameters[j], 1);
                }
            }
            return jacobMatrix;
        }//函数都可以用的雅可比矩阵,很慢

    }
}
