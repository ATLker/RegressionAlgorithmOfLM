import numpy as np
from scipy.misc import derivative

class RegressionAlgorithm():
    def __self__():
        """nothing to write here for just now,everything begins!!!"""
        pass
    def JacobMatrix(self,func,para,x):
        """That is what a genius do!!
        calculate the function's JacobiMatrix of parameters for each x
        func: is suppose to be func(para,x) in which the para represent a list of parameters
        para: refer to above
        x: a list of  x
         """
        datalength=len(x)#数据量
        paralength=len(para)#拟合的函数参数数目
        jacobMatrix=np.zeros((datalength,paralength))#创建一个全0矩阵
        #给矩阵赋值
        for i in range(datalength):
            for j in range(paralength):
                f=lambda k:func([para[n] if n!=j else k for n in range(paralength)],x[i])
                jacobMatrix[i,j]=derivative(f,para[j],dx=1.0e-6)
        return jacobMatrix
    def LM(self,func,x,y,para,lamda=0.5,iteration=50,increase=4,decrease=3):
        zhuangtai=1#减少计算量的状态判断数，在不更新参数时改为0减少下一次迭代计算量
        datalength=len(x)#数据量
        paralengh=len(para)#拟合的函数参数数目
        #循环开始
        for i in range(iteration):
            if zhuangtai==1:
                #计算误差向量
                ev=np.array([y[i]-func(para,x[i]) for i in range(datalength)])
                #数据总误差
                error=np.dot(ev,ev)
                #计算雅可比矩阵
                jacobMatrix=self.JacobMatrix(func,para,x)
                #计算黑塞矩阵用雅可比矩阵近似
                hessianMatrix=np.dot(np.transpose(jacobMatrix),(jacobMatrix))
            #H=J*J+lamda*I.  算法的灵魂所在!!!
            hessianMatrixPlus=hessianMatrix+lamda*np.identity(paralengh)
            #计算参数更新步长
            H_LL=np.linalg.inv(hessianMatrixPlus)
            J_LL=np.dot(np.transpose(jacobMatrix),ev)
            dp=np.dot(H_LL,J_LL)
            #暂时更新参数到para_Lm
            para_Lm=para+list(dp)
            #计算一下新参数误差
            ev_Lm=np.array([y[i]-func(para_Lm,x[i]) for i in range(datalength)])
            error_Lm=np.dot(ev_Lm,ev_Lm)
            #比一下和之前没更新时误差谁大
            if(error_Lm>error):
                #若误差增大，跳过本次更新，增大lamda，继续下一次寻找
                lamda*=increase
                zhuangtai=0
                continue
            #误差减小的话缩小Lamda,更新参数
            lamda/=decrease
            zhuangtai=1
            para=para_Lm
            #如果更新步长过小就结束
            if( np.max(abs(dp)) < 1.0e-3 ):
                break
        #返回更新后的参数
        return para
    def ElementReplace(list,i,value):
        """let list[i]=value ,then return the changed list"""
        length=len(list)
        list=[list[n] if n!=i else value for n in range(length)]
        return list

