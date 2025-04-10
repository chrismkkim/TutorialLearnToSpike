dt = 0.1 #simulation timestep (ms)

# training variables
nloop          = 20; #300
penlambda      = 0.05; # 0.1 or 0.5 
penlamEE       = 0.04; # 3.0
penlamEI       = 0.04; # 3.0
penlamIE       = 0.04; # 3.0
penlamII       = 0.04; # 3.0
penlamFF       = 1.0;
penmu          = 8.0; # 2.0
fracTrained    = 1.0;
learn_every    = 20.0
learn_step     = Int(learn_every/dt)

# innate, train, test time
train_duration = 2000.;
stim_on        = 800.;
stim_off       = 1000.;
train_time     = stim_off + train_duration;

Nsteps = Int(train_time/dt;)

# neuron param      
taum = 10; #membrane time constant for exc. neurons (ms)
taui = 10; 
thresh = 1.0 # spike threshold
threshi = 1.0   
refrac = 0.1 # refractory period
vre = 0.0

#synaptic time constants (ms) 
taus = 3
tauidecay = 3
taudecay_plastic = 150.0

# network size      
Ncells = 100;
Ne = Int(Ncells*0.5);
Ni = Int(Ncells*0.5);

# connectivity 
K = Ncells
pree = K/Ne
prei = K/Ne
prie = K/Ne
prii = K/Ne
sqrtK = sqrt(K)

# synaptic strength
g = 1.0 # 1.0, 1.5
je = 2.0 / sqrtK * taum * g
ji = 2.0 / sqrtK * taum * g 
jx = 0.12 * sqrtK * g 

jee = je*0.15
jie = je
jei = -ji*0.75
jii = -ji

muemin = jx*0.6 # exc external input
muemax = jx*0.6
muimin = jx*0.6 # inh external input
muimax = jx*0.6

# # plastic weights
# L = round(Int,sqrt(K)*L_list[ll]) # number of exc/inh plastic weights per neuron
# Lexc = L # excitatory L
# Linh = L # inhibitory L
# wpscale = sqrtK * 0.5

L = round(Int,sqrt(K))*2 # number of exc/inh plastic weights per neuron
Lffwd = 10 #round(Int, L/2) # feedfoward L/2
Lexc = L # excitatory L
Linh = L # inhibitory L
wpscale = sqrtK * 2 / 4.0

wpee = je * sqrtK / wpscale
wpie = je * sqrtK / wpscale
wpei = -ji * sqrtK / wpscale * (6.5/12.0) # initial network: exc rate = 6.5 Hz, inh rate = 12.0 Hz
wpii = -ji * sqrtK / wpscale * (6.5/12.0) # initial network: exc rate = 6.5 Hz, inh rate = 12.0 Hz

wpffwd = wpee * 2.0

maxrate = 500 #(Hz) maximum average firing rate.  if the average firing rate across the simulation for any neuron exceeds this value, some of that neuron's spikes will not be saved


mutable struct paramType
    train_duration::Float64
    nloop::Int64
    penlambda::Float64
    penlamEE::Float64
    penlamEI::Float64
    penlamIE::Float64
    penlamII::Float64
    penlamFF::Float64
    penmu::Float64
    fracTrained::Float64
    learn_every::Float64
    learn_step::Int64
    stim_on::Float64
    stim_off::Float64
    train_time::Float64
    dt::Float64
    Nsteps::Int64
    Ncells::Int64
    Ne::Int64
    Ni::Int64
    pree::Float64
    prei::Float64
    prie::Float64
    prii::Float64
    taum::Float64
    taui::Float64
    K::Int64
    sqrtK::Float64
    L::Int64
    Lffwd::Int64
    Lexc::Int64
    Linh::Int64
    wpscale::Float64
    je::Float64
    ji::Float64
    jx::Float64
    jee::Float64
    jei::Float64
    jie::Float64
    jii::Float64
    wpee::Float64
    wpei::Float64
    wpie::Float64
    wpii::Float64
    wpffwd::Float64
    muemin::Float64
    muemax::Float64
    muimin::Float64
    muimax::Float64
    vre::Float64
    thresh::Float64
    threshi::Float64
    refrac::Float64
    taus::Float64
    tauidecay::Float64
    taudecay_plastic::Float64    
    maxrate::Float64
end

p = paramType(train_duration,nloop,penlambda,penlamEE,penlamEI,penlamIE,penlamII,penlamFF,penmu,fracTrained,learn_every,learn_step,stim_on,stim_off,train_time,dt,Nsteps,Ncells,Ne,Ni,pree,prei,prie,prii,taum,taui,K,sqrtK,L,Lffwd,Lexc,Linh,wpscale,
je,ji,jx,jee,jei,jie,jii,wpee,wpei,wpie,wpii,wpffwd,muemin,muemax,muimin,muimax,vre,thresh,threshi,refrac,taus,tauidecay,taudecay_plastic,maxrate);
  


