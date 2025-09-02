import numpy as np

class Tensor(object):
    def __init__(self , data ,autogrd = False ,  crs =  None , cr_op = None , id = None):
        self.data = np.array(data)
        self.crs = crs
        self.cr_op = cr_op
        self.grd =None

        self.autogrd = autogrd 
        self.chld = {}
        if(id is None ):
            id = np.random.randint(0 , 100000000000)
        self.id = id 

        if(crs is not None ):
            for i in crs :
                if(self.id not in i.chld ):
                    i.chld[self.id] = 1
                else:
                    i.chld[self.id] += 1
        
    def all_chld_grd_acced_for(self):
        for id , cnt in self.chld.items():
            if(cnt != 0):
                return False
            
        return True

    ################################
    #       MATH
    ################################

    def __add__ (self, other) :
        if(self.autogrd and other.autogrd ):
            return Tensor(self.data + other.data , crs = [self , other] , cr_op='add' , autogrd=True)
        return Tensor(self.data + other.data)
    
    def __neg__(self  ):
        if(self.autogrd):
            return Tensor(self.data *-1 , autogrd=True , crs=[self] ,cr_op="neg" )
        return Tensor(self.data * -1)
    
    def __sub__(self , other):
        if(self.autogrd and other.autogrd):
            return  Tensor(self.data -  other.data , autogrd=True , crs=[self , other] , cr_op="sub" )
        return Tensor(self.data - other.data)

    def __mul__(self , other):
        if(self.autogrd and other.autogrd ):
            return Tensor(self.data * other.data , autogrd=True , crs=[self, other] , cr_op="mul" )
        return Tensor(self.data * other.data)
    
    def sum(self , dim):
        if(self.autogrd):
            return Tensor(self.data.sum(dim) ,autogrd=True , crs=[self] , cr_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))
    
    def expand(self , dim , copies):
        tr_cmd = list(range(0 , len(self.data.shape)))
        tr_cmd.insert(dim , len(self.data.shape ))
        nw_shape = list(self.data.shape) + [copies]
        nw_data = self.data.repeat(copies).reshape(nw_shape)
        nw_data =nw_data.transpose(tr_cmd)

        if(self.autogrd):
            return Tensor(nw_data , autogrd=True , crs=[self] , cr_op="expand_"+str(dim) )
        return Tensor(nw_data)
    
    def transpose(self):
        if(self.autogrd):
            return Tensor(self.data.transpose() ,autogrd=True , crs=[self] , cr_op="transpose" )
        return Tensor(self.data.transpose() )
    
    def mm(self , x):
        if(self.autogrd):
            return Tensor(self.data.dot(x.data) , autogrd=True , crs=[self , x] , cr_op="mm")
        return Tensor(self.data.dot(x.data))

    def sigmoid(self):
        if(self.autogrd):
            return Tensor(1/(1+np.exp(-self.data)) , autogrd=True , crs = [self] , cr_op="sigmoid" )
        return Tensor(1/(1+np.exp(-self.data)))
    
    def tanh(self):
        if(self.autogrd):
            return Tensor(np.tanh(self.data) , autogrd= True , crs=[self] , cr_op="tanh" )
        return Tensor(np.tanh(self.data))
    
    def relu(self):
        if(self.autogrd):
            return Tensor(((self.data >= 0) * self.data ) , autogrd=True , crs=[self] , cr_op="relu")
        return Tensor(((self.data >= 0) * self.data ) )
    
    def index_select(self , indices):
        if(self.autogrd):
            nw = Tensor(self.data[indices.data] , autogrd=True , crs=[self] , cr_op="index_select")
            nw.index_select_indices = indices
            return nw
        return Tensor(self.data[indices.data])
    
    def cross_entropy(self, target_indices ):
        tmp = np.exp(self.data)
        sftmx_out = tmp / np.sum(tmp , axis=len(self.data.shape)-1 , keepdims=True)
        t = target_indices.data.flatten()
        p = sftmx_out.reshape(len(t), -1)
        trg_dis = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (trg_dis)).sum(1).mean()

        if (self.autogrd):
            out = Tensor(loss , autogrd=True , crs=[self] , cr_op="cross_entropy")
            out.softmax_output = sftmx_out
            out.target_dist = trg_dis
            return out
        return Tensor(loss)
    #################################
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())
    
    
    def back(self, grd , grd_origrn = None ):
        ##self.grd = grd 
        if(self.autogrd):
            if(grd_origrn is not None):
                if(self.chld[grd_origrn.id] == 0):
                    raise Exception("self.chld[grd_origrn.id] == 0")
                else:
                    self.chld[grd_origrn.id] -= 1

            if(grd is None):
                grd = Tensor(np.ones_like(self.data))

            if(self.grd is None):
                self.grd = grd
            else:
                self.grd +=grd
            
            if(self.crs is not None  and (self.all_chld_grd_acced_for() or (grd_origrn is None))):
                if(self.cr_op =="add"):
                    self.crs[0].back(self.grd , self )
                    self.crs[1].back(self.grd , self )
                if(self.cr_op == "neg"):
                    self.crs[0].back(self.grd.__neg__())
                if(self.cr_op == "sub"):
                    self.crs[0].back(Tensor(self.grd.data) , self)
                    self.crs[1].back(self.grd.__neg__().data)
                if(self.cr_op == "mul" ):
                    self.crs[0].back(self.grd * self.crs[1] , self)
                    self.crs[1].back(self.grd * self.crs[0] , self)
                
                if(self.cr_op =="mm"):
                    act = self.crs[0]
                    w = self.crs[1]
                    nw= self.grd.mm(w.transpose())
                    act.back(nw)
                    w.back(self.grd.transpose().mm(act).transpose())
                
                if(self.cr_op == "transpose"):
                    self.crs[0].back(self.grd.transpose())
                
                if("sum" in self.cr_op):
                    dim = int(self.cr_op.split("_")[1])
                    ds = self.crs[0].data.shape[dim]
                    self.crs[0].back(self.grd.expand(dim , ds))
                
                if("expand" in self.cr_op):
                    dim = int(self.cr_op.split("_")[1])
                    self.crs[0].back(self.grd.sum(dim))
                if(self.cr_op == "sigmoid"):
                    ones = Tensor(np.ones_like(self.grd.data) )
                    self.crs[0].back(self.grd *(self * (ones - self)))
                if(self.cr_op == "tanh"):
                    ones = Tensor(np.ones_like(self.grd.data) )
                    self.crs[0].back(self.grd *(ones  - (self * self)))
                ###WARNING :
                if(self.cr_op =="relu"):
                    self.crs[0].back((self.data > 0) * self.grd.data)
                if(self.cr_op == "index_select" ):
                    nw_grd = np.zeros_like(self.crs[0].data)
                    indices_ = self.index_select_indices.data.flatteen()
                    grd_ = grd.data.reshape(len(indices_) , -1 )
                    for i in range(len(indices_)):
                        nw_grd[indices_[i]]+=grd[i]
                    self.crs[0].back(Tensor(nw_grd))
                if(self.cr_op =="cross_entropy"):
                    dx = self.softmax_output - self.target_dist
                    self.crs[0].back(Tensor(dx))

                
class SGD(object):
    def __init__(self , parameters , alpha = 0.01):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0
    def step(self ,zero = True):
        for p in self.parameters :
            p.data -= p.grad.data * self.alpha
            if(zero):
                p.grad.data *= 0
        

class Layer(object):
    def __init__(self):
        self.parameters = list()

    def get_par(self):
        return self.parameters
    
class Linear(Layer):
    def __init__(self , n_inp , n_out):
        super().__init__()
        w = np.random.randn(n_inp , n_out) * np.sqrt(2.0 / (n_inp)) 
        self.w = Tensor(w , autogrd=True)
        self.b = Tensor(np.zeros(n_out) , autogrd=True)

        self.parameters.append(self.w)
        self.parameters.append(self.b)   

    def forward(self , inp):
        return  inp.mm(self.w) +self.b.expand(0, len(input.data))


class Sequent(Layer):
    def __init__(self , layers = list()):
        super().__init__()

        self.layers = layers
    
    def add(self , layer):
        self.layers.append(layer)
    def forward(self , input):
        for l in self.layers:
            input = l.froward(input)
        return input
    def get_pars(self):
        p = list()
        for l in self.layers :
            p+= l.get_par()
        return p

class MSELoss(Layer):
    def __init__(self):
        super().__init__()
    def forward(self , pred , target):
        return((pred - target)*(pred - target)).sum(0)
    
class Tanh(Layer):
    def __init__(self):
        super().__init__()
    def forward(self , input):
        return input.tanh()
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
    def forward(self , input):
        return input.sigmoid()

class Embedding(Layer):
    def __init__(self , vocab_size , dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.w = Tensor(((np.random.rand(vocab_size, dim) - 0.5) / dim) , autogrd=True)
        self.parameters.append(self.w)

    def forward(self, input):
        return self.w.index_select(input)
    
    
class CrossEntropyLoss(object):
    def __init__(self):
        super().__init__()

    def forward(self , input , target):
        return input.cross_entropy(target)

class RNNCell(Layer):
    def __init__(self , n_inp , n_hidden , n_out , activation='sigmoid'):
        super().__init__()
        self.n_inp = n_inp
        self.n_hid =  n_hidden
        self.n_out = n_out

        if activation =='sigmoid':
            self.activation = 'sigmoid'
        elif activation == 'tahn':
            self.activation = 'tahn'
        else:
            raise Exception("elif activation == 'tahn':  self.activation = 'tahn' else:")
        self.w_ih = Linear(n_inp , n_hidden)
        self.w_hh = Linear(n_hidden , n_hidden)
        self.w_ho = Linear(n_hidden , n_out)

        self.parameters +=self.w_ih.get_par()
        self.parameters +=self.w_hh.get_par()
        self.parameters +=self.w_ho.get_par()

    def forward(self , input , hidden ):
        fph = self.w_hh.forward(hidden)
        com = self.w_ih.forward(input) + fph
        nw_hid = self.activation.forward(com)
        out = self.w_ho.forward(nw_hid)
        return out , nw_hid
    def init_hidden(self , batch_size = 1):
        return  Tensor(np.zeros((batch_size ,self.n_hid)) , autogrd=True)
    



