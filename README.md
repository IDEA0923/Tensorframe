this is my simple frame work 

how use ( simple example ):


    from framework import Tensor

    #print(str(framework.Tensor([1 , 2 ,3])))

    l0 = Tensor([1 , 2 , 3] , autogrd=True)
    l1 = Tensor([[1 , 5 , 3] , [1 , 5 , 3] , [1 , 5 , 3]] , autogrd=True)
    l2 = Tensor([1 , 2 , 4] , autogrd=True)


    a =  l0.mm(l1)
    b =  a.sum(0)
    c = b.mm(l2)

    print(str(c.data))

    c.back(Tensor([1 , 1 , 1]))

    print(str(l0.grd))							
