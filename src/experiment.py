from typing import Dict, Sequence, Any
import numpy as np
from action_scheduler import Schedule
from ResistorNetwork import ResistorNetwork
from tqdm import tqdm

class Experiment:
    """
    Base scaffold for an experiment run.
    
    Attributes
    ----------
    required_params : Sequence[str]
        List of parameter names that must be provided in `params`.
    schedule : Any
        Object defining the order of actions.
    DATA : np.ndarray
        Array of shape (#train + #test, #inputs + #outputs).
    IONODE : np.ndarray
        Array of shape (#inputs + #outputs,) defining node locations.
    """
    # Define here the names of all required params
    required_params: Sequence[str] = [
        "ALF", # learning rate
        "ETA", # nudge factor
        "OVC", # overclamping
        "NOR", # batch normalization (1) or adam optimizer (2)
        "DIF", # differential outputs
        "DGC", # dG (gradient) clipping
        "TRA", # number of training datapoints (also implicitly # test)
        "INP", # number of inputs (also implicitly # outputs)
        "TDS", # training data selection (0 for random, < 0 to indicate cycling with period abs(n))
        "GMN", # minimum for learning parameters
        "GMX", # maximum for learning parameters
        "FLP", # multiply this to learning steps -- fixes resistance vs capacitance
        "EPS", # for LOS==2 (Q Learning), starting fraction that a random action (datapoint) is taken (trained)
        "EPM", # for LOS==2 (Q Learning), ending fraction that a random action (datapoint) is taken (trained)
        "GAM", # for LOS==2 (Q Learning) with env state, discount factor for future estimates
        "LOS", # loss function - 0 for MSE, 1 for shifted hinge, 2 for Q learning
      #  "VMX", # maximum voltage (for use in classification loss/labeling)
      #  "VMN", # minimum voltage (for use in classification loss/labeling)
      #  "CEA", # Cross-Entropy alpha: modulates crossentropy loss magnitude to keep clamping within range
        "UDE", # update exponent
        "MXP", # MULT exponent (for overclamping)
    ]
    # to add: EQP for eqprop, 
       

    def __init__(
        self,
        params: Dict[str, Any],
        schedule: Any,
        data: np.ndarray,
        ionode: np.ndarray,
        network: ResistorNetwork
    ):


        self.beta1 = 0.6
        self.beta2 = 0.999
        self.epsilon = 1e-3
        
        self.m = np.zeros_like(network.params)
        self.v = np.zeros_like(network.params)

        self.VBD = [np.min(data),np.max(data)]
        
        # 1. Check for missing params
        missing = set(self.required_params) - set(params.keys())
        if missing:
            raise ValueError(f"Missing required params: {missing}")

        # 1. Assign params to attributes
        for name in self.required_params:
            setattr(self, name, params[name])

        # 2. Store the schedule object
        self.SCHEDULE = schedule
        self.STP = self.SCHEDULE.total_steps

        # 3. Store the data array
        #    Expect shape: (#train + #test, #inputs + #outputs)
        if not isinstance(data, np.ndarray):
            raise TypeError("`DATA` must be a numpy.ndarray")
        self.DATA = data
        self.TST = np.shape(self.DATA)[0]-self.TRA
        self.OUT = np.shape(self.DATA)[1]-self.INP
        self.MULTS = np.ones(self.TRA)/self.TRA

        # 4. Store the ionode array
        #    Expect shape: (#inputs + #outputs,)
        if not isinstance(ionode, np.ndarray):
            raise TypeError("`IONODE` must be a numpy.ndarray")
        self.IONODE = ionode
        if np.size(self.IONODE) != self.INP+self.OUT:
            raise SizeError("IONODE must have INP+OUT values, the same as axis 1 of DATA")

        self.t = -1 # not yet trained
        self.batch_step = 0

        # 5. Store the network object
        self.network = network
        self.NN = network.num_nodes
        self.NE = np.size(network.node_from)

        self.update = network.params*0.0 # store spot for learning updates

        self.stored_states = []
        self.stored_clamp_states = []
        self.stored_params = []
        self.stored_state_steps = []
        self.stored_param_steps = []
        self.stored_loss = []
        self.stored_reward = []
        self.stored_reward_std = []
        self.stored_reward_count = []
        self.rewards=[]

        self.freestates = np.zeros((self.TRA+self.TST,self.NN))
        self.clampstates = np.zeros((self.TRA,self.NN)) # no clamped state for test points

        self.sincebatch_free = np.zeros(self.TRA)
        self.sincebatch_clamped = np.zeros(self.TRA)
        
        self.QReward = None
        self.QState = None
        self.kill = False

    """ Runs the experiment! Simply iterate through actions, update internal clock"""
    def run(self,kill=None,msg='Training'):
        
        for act in self.SCHEDULE.get_actions(0):
            self.perform_action(act,0)          
        self.t = 0
        
        for t in tqdm(range(1,self.SCHEDULE.total_steps),desc=msg,total=self.SCHEDULE.total_steps):
            for act in self.SCHEDULE.get_actions(t):
                self.perform_action(act,t,kill=kill)          
            self.t = t+1
            if self.kill:
                break

    """ calculate loss for given datapoints (or their mean) based on the EXP's loss function"""
    def loss(self,datapoints=None,mean=False): 
        
        if datapoints is None:
            datapoints = np.arange(self.TRA+self.TST) # default is every datapoint

       
        outvals = self.freestates[np.ix_(datapoints, self.IONODE[self.INP:])]
        labels = self.DATA[np.ix_(datapoints,np.arange(self.INP,self.INP+self.OUT))]
 
        if self.DIF:
            outvals = outvals[:,::2]-outvals[:,1::2]
            labels = labels[:,::2]-labels[:,1::2]
        
        if self.LOS == 0: # MSE
            LOSS = np.sum((outvals-labels)**2,axis=1)
        
        elif self.LOS == 1: # Shifted Hinge
            #raise Exception('problem!')
            #LOSS = self.cross_entropy_loss(u=outvals, y=labels) 
            LOSS = np.sum((outvals-labels)**2,axis=1)
            for idx in range(len(LOSS)):
                if (np.sign(outvals[idx]-labels[idx]) == np.sign(labels[idx])):
                    LOSS[idx]  = 0


        elif self.LOS == 2: # Q Learning (discrete) -- will need something else for continuous
            LOSS = np.sum((outvals-labels)**2,axis=1)

        if mean:
            LOSS = np.mean(LOSS)

        return LOSS
        
    """ Let's actually perform these actions... """
    def perform_action(self,act,t,kill=None):

        if act == 'INIT':
            print('DOING INIT!!')
        elif act == 'TRAIN':

            if self.LOS==2:
                # step is determined from TDS value
                self.perform_train_step(self.TDS,t)
                                    
            else: # standard supervised training
                # Choose datapoint
                if self.TDS == 0:
                    train_idx = np.random.choice(self.TRA)
                elif self.TDS < 0: # corrected 5/21/25
                    train_idx = int(t//(-self.TDS)) % self.TRA
                else:
                    raise ValueError("TDS positive has no coded meaning")
                
                self.perform_train_step(train_idx)
            
        elif act == 'MEASURE': 
            datapoints = np.arange(self.TRA+self.TST) # all datapoints
            nodes = np.arange(self.NN) # every node
            values = self.measure(datapoints,nodes,clamp=False)  # measure (equilibrate)
            clampvalues = self.measure(datapoints,nodes,clamp=True)  # measure (equilibrate)
            self.stored_states.append(values) # store it
            self.stored_clamp_states.append(clampvalues) # store it
            self.stored_state_steps.append(t)
            self.stored_loss.append(self.loss(mean=False)) # store training loss
            if len(self.rewards)>0:
                self.stored_reward.append(np.mean(self.rewards))
                self.stored_reward_std.append(np.std(self.rewards))
                self.stored_reward_count.append(np.size(self.rewards))
            else:
                self.stored_reward.append(np.nan)
                self.stored_reward_std.append(np.nan)
                self.stored_reward_count.append(0)
            self.rewards = []
                
            if kill == 'CLASSERR0':
                if np.sum(self.stored_loss[-1]>=np.squeeze(np.diff(self.DATA[:,self.INP:],axis=1))**2)==0:
                    self.kill = True
        
        elif act == 'STOREPARAM': 
            self.store_params(t=t)  # measure it!
            
        # Apply stored update to params
        elif act == 'BATCH':
            self.apply_batch()
            
        else:
            raise ValueError("Bad Action Given: "+act)

    """ apply batched updates """
    def apply_batch(self):    
       # if G_update_scaling:
       #     update *= (network.params-GMN)

        if self.NOR==2:
            self.batch_step+=1
            self.m = self.beta1 * self.m + (1 - self.beta1) * (-self.update)
            self.v = self.beta2 * self.v + (1 - self.beta2) * (self.update**2)
    
            m_hat = self.m / (1 - self.beta1 ** self.batch_step)
            v_hat = self.v / (1 - self.beta2 ** self.batch_step)

            self.update = -m_hat / (np.sqrt(v_hat) + self.epsilon)

        if np.any(np.isnan(self.update)):
            print("nan update",self.update)
            print(self.m,self.v,self.network.params)
            raise Exception('prob')
            self.update = np.zeros_like(self.update)

        if self.DGC !=0:
            self.update = np.clip(self.update,-self.DGC,self.DGC)
            
        self.network.params = np.clip(self.network.params+self.update,self.GMN,self.GMX)
        self.update *=0
        self.sincebatch_free*=0
        self.sincebatch_clamped*=0


        
    """ take specified datapoints and nodes, store them """ 
    def measure(self,datapoints,nodes,clamp=False):
        for idx in datapoints:
            if clamp:
                self.clampstates[idx] = self.getState(idx,free=False) # equilibrate state
            else:
                self.freestates[idx] = self.getState(idx) # equilibrate state

        if clamp:
            return self.clampstates[np.ix_(datapoints,nodes)]
        else:
            return self.freestates[np.ix_(datapoints,nodes)]

    """ store learning parameters """ 
    def store_params(self,t):
        self.stored_params.append(np.array(self.network.params)) # save it    
        self.stored_param_steps.append(t)

    
    """ enforce inputs, equilibrate, measure dV, clamp, equilibrate, measure dV, calc learn time, store updates """
    def perform_train_step(self,train_idx,overall_step=None):

        # find equilibrium free state, store delta V's
        self.freestates[train_idx] = self.getState(train_idx)
        doUpdate = True
        
        if self.LOS==2: # Q learning!
            frac = overall_step/self.STP
            if np.random.rand()<((self.EPM*frac) + self.EPS*(1-frac)): # we choose a random action
                action_idx = np.random.choice(self.OUT//(1+self.DIF))
              
            else: # we choose the optimal action -- 
                outputs = np.array(self.freeOut(train_idx))
                if self.DIF>0:
                    outputs = outputs[::2]-outputs[1::2]
                    
                action_idx = np.argmax(outputs)
                

            outnode = self.IONODE[self.INP+action_idx]

            reward = self.QReward(train_idx,action_idx) # state, action
            
            self.rewards.append(reward)

            new_env_state = self.QState(train_idx,action_idx) # let environment react
            
            if new_env_state != train_idx: #otherwise already solved for
                self.freestates[new_env_state] = self.getState(new_env_state) # store response
                # there are some efficiency gains to be made here avoiding redundant calcuations


            reward += self.GAM*(np.max(self.freeOut(new_env_state))-np.mean(self.freeOut(new_env_state)))
                                            
            clampOut = np.array(self.freeOut(train_idx))

            clampval = self.getClamps(clampOut[action_idx:action_idx+1], np.atleast_1d(reward))
            clampOut[action_idx] = clampval[0]
            
            #nodes = np.hstack([self.IONODE[:self.INP],outnode])
            voltages = np.hstack([self.DATA[train_idx,:self.INP],clampOut])
           
            
            if self.sincebatch_clamped[train_idx]:
                pass
            else:
                self.clampstates[train_idx] = self.network.solve_equilibrium(self.IONODE, voltages,initial_guess=self.clampstates[train_idx],usebounds=self.VBD)



        elif self.LOS==1: # shifted hinge
            O = np.diff(self.freeOut(train_idx))
            L = np.diff(self.DATA[train_idx,self.INP:])
            if np.sign(O-L) == np.sign(L):
                doUpdate = False
                self.clampstates[train_idx] = np.array(self.freestates[train_idx])

            else: 
            # find equilibrium clamped state for this datapoint
                self.clampstates[train_idx] = self.getState(train_idx,free=False)

        else: # MSE
            self.clampstates[train_idx] = self.getState(train_idx,free=False)

        
        
        if doUpdate: # currently always true
            Vf = np.array([self.freestates[train_idx][x]-self.freestates[train_idx][y] for x,y in zip(self.network.node_from, self.network.node_to)])
            Vc = np.array([self.clampstates[train_idx][x]-self.clampstates[train_idx][y] for x,y in zip(self.network.node_from, self.network.node_to)])
            
            # calculate learning (time) multiple
            if not self.OVC is None:
                if self.DIF and self.OUT !=2:
                    raise Exception('this isnt coded right for DIF')
                elif self.DIF:
                     self.MULTS[train_idx] = abs(np.diff(self.freeOut(train_idx)-self.DATA[train_idx,self.INP:]))
                else:
                    self.MULTS[train_idx] = np.linalg.norm(np.array([self.freestates[train_idx][i]-self.DATA[train_idx][num+self.INP] for num,i in enumerate(self.IONODE[self.INP:])]))

               
            if self.NOR==1:
                SCALE = self.MULTS[train_idx]/np.sum(self.MULTS**self.MXP)
            else:
                SCALE = 1.0
            
            # learn! (store update) ## this currently divides by ETA...
            self.update += ((self.network.params-self.GMN)**(self.UDE)) * \
                self.FLP * SCALE * (self.MULTS[train_idx] ** self.MXP) * \
                self.ALF * (Vf**2-Vc**2)  / self.ETA
        
        else: # not updating -- MULT should be 0
            self.MULTS[train_idx] = 0
            

        if self.LOS == 2: # update environment state!
            self.TDS = new_env_state
            

    """ return STORED free outputs """
    def freeOut(self,train_idx):
        return [self.freestates[train_idx][i] for i in self.IONODE[self.INP:]]

    """ SOLVE and return entire equilibrated state """
    def getState(self, train_idx,free=True):
    
        if free: 
            endidx = self.INP
        else:
            endidx = self.INP+self.OUT
        
        nodes = self.IONODE[:endidx]
        voltages = np.array(self.DATA[train_idx,:endidx])

        if not free: # change from labels to actual clamp values
            voltages[self.INP:] = self.getClamps(freeOut=self.freeOut(train_idx),labels = self.DATA[train_idx,self.INP:])

        if free:
            initial_guess = self.freestates[train_idx]
            if self.sincebatch_free[train_idx]:
                return initial_guess
            else:
                self.sincebatch_free[train_idx] = True
        else:
            initial_guess = self.clampstates[train_idx]
            if self.sincebatch_clamped[train_idx]:
                return initial_guess
            else:
                self.sincebatch_clamped[train_idx] = True
       

        return self.network.solve_equilibrium(nodes, voltages,initial_guess=initial_guess,usebounds=self.VBD)


    """ Generate clamping values given free outputs and labels (value to train towards)"""
    def getClamps(self,freeOut, labels):
      
        if self.LOS==1: # pick label for DIF + HINGE
            if self.DIF != 1 or self.OUT != 2:
                raise Exception('Hinge loss only stipulated for single dif output!')

           # clampouts = self.dynamic_ce_label(u=freeOut,y=labels)
        # keep just the labels in DATA
        
        if self.OVC is None:
            clampouts = [i for i in labels]

        else: # if overclamping, switch to OC values based on sign of error
            if self.DIF:
                clampouts = []
                doublediff = np.diff(np.array(freeOut)-np.array(labels))
                for idx in range(len(doublediff)):
                    if np.sign(doublediff[idx])<0:
                        clampouts+= self.OVC
                    else:
                        clampouts+= self.OVC[-1::-1] 
            else:
                clampouts = [self.OVC[0] if np.sign(self.OVC[0]-f)==np.sign(L-f) else self.OVC[1] for (f,L) in zip(freeOut,clampouts)]

        # apply ETA
        clamps = [f*(1-self.ETA) + self.ETA*L for (f,L) in zip(freeOut,clampouts)]

        if self.DIF: # correct clamping to mirror mean of free output
            for t in range(self.OUT//2):
                clamps[t*2:t*2+2] += self.ETA* (-np.mean(clamps[t*2:t*2+2])+np.mean(freeOut[t*2:t*2+2]))

        return clamps


        
    def dynamic_ce_label(self,
                     u: np.ndarray,
                     y: np.ndarray,
                     eps: float = 1e-7
                    ) -> np.ndarray:
        """ Compute dynamic MSE‐labels z so that
          (g - z) * dg/du = alpha * (f - y0),
        where g = VMN + R*f, f=sigmoid(u), y0=(y-VMN)/R.
        """
        # in [eps,1-eps]
        f0 = np.clip(1/(1 + np.exp(-np.array(u))), eps, 1 - eps)
        # normalize y into [0,1]
        y0 = (y - self.VMN) / (self.VMX - self.VMN)

        # SCALE the correction by alpha:
        z0 = f0 - self.CEA * (f0 - y0) / (f0 * (1 - f0))

        # back into [VMN,VMX]
        return self.VMN + (self.VMX - self.VMN) * z0


    def cross_entropy_loss(self,
                       u: np.ndarray,
                       y: np.ndarray,
                       eps: float = 1e-12
                      ) -> np.ndarray:
        """
        Returns per-sample CE loss L = -sum(y0 * log p).
        """
        # normalize targets into [0,1]
        y0           = (y - self.VMN) / (self.VMX-self.VMN)
    
        # stabilized softmax
       # shift     = u - np.max(u, axis=1, keepdims=True)
       # exp_shift = np.exp(shift)
       # p         = exp_shift / exp_shift.sum(axis=1, keepdims=True)
    
        exp_shift = np.exp(u - np.max(u, axis=1, keepdims=True))
        p         = exp_shift / exp_shift.sum(axis=1, keepdims=True)

        # cross-entropy: -sum over classes of y0 * log p
        return -np.sum(y0 * np.log(p + eps), axis=1)

    
    def cross_entropy_error(self,
                            u: np.ndarray,
                            y: np.ndarray
                           ) -> np.ndarray:
        """
        Compute delta = alpha*(p - y0), with p=softmax(u), y0=(y-VMN)/R.
        """
        # normalize y
        y0 = (y - self.VMN) / (self.VMX - self.VMN)

        # stabilized softmax
        shift     = u - np.max(u, axis=1, keepdims=True)
        exp_shift = np.exp(shift)
        p         = exp_shift / exp_shift.sum(axis=1, keepdims=True)

        # shrink the CE‐gradient by alpha
        return self.CEA * (p - y0)
        
    