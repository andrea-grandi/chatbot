Analysing honeybees’ division of labour in broodcare by a multi-agent model

Thomas Schmickl1,2 and Karl Crailsheim1,3

1Department for Zoology, Karl-Franzens-University Graz, Universit¨atsplatz 2, A-8010 Graz
2thomas.schmickl@uni-graz.at
3karl.crailsheim@uni-graz.at

Abstract

We describe a multi-agent model of a honeybee colony and
show several applications of the model that simulate exper-
iments that have been performed with real honeybees. Our
special emphasis was on the decentralized, self-organized
regulation of brood nursing, which we successfully simu-
lated: We found that brood manipulations, food-deprivation
experiments and colony-size manipulations can be explained
by the mechanisms we implemented into our model described
here. Our agents can perform various tasks (foraging, storing,
nursing). The model is spatially resolved, and contains a des-
ignated broodnest area as well as a designated honey/nectar
storage area. All bees (and larvae) consume nectar/honey
at a task-speciﬁc rate, allowing us to track the ﬂow of nec-
tar through the colony. Several kinds of stimuli, which are
important for division of labour, were modelled in detail:
dances, contact stimuli and chemical signals.

Introduction
The ability of social insects to divide the colony’s work via
specialisation, polyethism and task partitioning has fasci-
nated scientists since decades. For example, early work of
(Lindauer, 1952; R¨osch, 1952; Sakagami, 1953) described
the impressive ability of honeybees to specialize on different
tasks based on an age-based scheme (temporal polyethism).
In recent years, several conceptual models have been pro-
posed to explain the basic proximate mechanisms that lead
to division of labour in social insects in general, see Beshers
and Fewell (2001) for a review, and for honeybees in de-
tail: Age-based polyethism (Seeley, 1982; Johnson, 2005),
regulation by queuing delays (Seeley, 1992), foraging for
work theory (Franks and Tofts, 1994), threshold reinforce-
ment (Theraulaz et al., 1998), and social inhibition (Beshers
et al., 2001). Many of these concepts were also investigated
by mathematical models and computer simulation (Ander-
son, 1998; Gautrais et al., 2002). One the one hand, these
models focused very well on the speciﬁc key process that
they were built to examine, on the other hand, they lack
many speciﬁc details that are signiﬁcantly affecting the be-
haviour of social insects. To ﬁll this gap and to allow speciﬁc
simulation of honeybees’ division of labour, we constructed
a multi-agent model of a honeybee colony that builds on the

Artificial Life XI 2008

 529

Figure 1: Typical screenshot of our multi-agent simulation
at run-time. Bee agents move across the hive space and, de-
pending on their history, emit several sorts of stimuli: wag-
gle/tremble dances and offering signals. Hungry larvae also
emit chemical hunger stimuli, which diffuse in the central
broodnest area. Unemployed bees can react to all of these
stimuli and switch to one of the modelled task cohorts.

ideas of the before mentioned models and incorporates sev-
eral important honeybee speciﬁc details:

1. A typical spatial distribution of brood and food in the hive.

2. Complex behavioural programs of specialized workers.

1. All patches update their status (decay of chemicals).

3. Characteristics of the spreading of different kinds of stim-

2. All agents emit stimuli, chemicals are diffused.

uli (chemicals, sounds/vibration, light).

4. Agents physiology (energetic expenditures).

5. Flow of nutrients among the agents and the combs.

Our multi-agent model (named TaskSelSim) is imple-
mented in NetLogo (Wilensky, 1999). The implementation
of the model (equations, parameter values) have been de-
scribed in detail in (Schmickl and Crailsheim, 2008b). In
this article, we describe the models implementation in a
lower degree of details and concentrate on those details that
are important for the focal questions described here: How
does the brood status affect the division of labour in the sim-
ulated honeybee colony and how does the colony status af-
fect the brood nursing. Other aspects of division of labour
(effects of selective removals/additions of task cohorts) were
already investigated in (Schmickl and Crailsheim, 2008a),
thus we did not perform such experiments in the study pre-
sented here.

Brood nursing (feeding brood with honey, pollen and
pollen derived gland products) is a distributed process in a
honeybee colony: Each specialised nurse bee feeds many
larvae sequentially and each larva is fed by many nurses.
The brood is allocated in a central area in the hive, one
larva is occupying one comb cell. We studied the nurs-
ing of brood in honeybees with several ethological studies,
see Schmickl and Crailsheim (2002). These experiments
showed that brood nursing is regulated in a homoeostatic,
adaptive way.
It was shown by (Huang and Otis, 1991b)
that nurse bees preferentially inspect comb cells that are oc-
cupied by larvae and that artiﬁcially starved larvae receive
preferential nursing (Huang and Otis, 1991a). The hunger
state of a larva is communicated to nurse bees by emission of
chemical substances (pheromones). All of these facts were
incorporated in our model (together with an implementation
of the foraging process and the nectar storing process), to
generate a model that is able to integrate many (separately
derived) hypothesises of honeybees’ regulation of division
of labour into one single consistent process.

The Model
Our model depicts one honeybee colony consisting of agents
(adult bees, larvae), stimuli (dances, contact stimuli, chem-
ical signals, light) and resource stores (nectar and honey is
used synonymously). The hive space is modelled in discrete
patches (31 x 52) but the adult bee agents can move across
these patches in continuous motion. The intensity of local
stimuli is modelled discrete, following the grid of patches
that represent the comb cells. Figure 1 depicts the typical
spatial distribution of agents, stimuli and resource stores.

Within each time step, the following functions are exe-

cuted iteratively:

3. All agents consume nectar.

4. All adult agents decide to engage or to give up a task.

5. All adult agents perform behaviour according to their

task.

Modelled Tasks
Depending on the task the bees are engaged in, they perform
the following behavioural programs:

Unemployed bees: These bees move randomly in the hive.
In our model, bees had to switch to this unemployed state
at least for one time step before they could engage in a
different task.

Forager bees: These bees leave the hive with a low (but
sufﬁcient) crop load. They ﬂy to the nectar source, ﬁll
up their crop and ﬂy back to the entrance. There they
emit the unloading stimulus to attract nearby storage bees
which take over the nectar load. After some time of ran-
dom movement in the hive, they can perform a waggle or
a tremble dance (see below for more details). Afterwards,
they leave the hive again towards the nectar source.

Storer bees: These bees wait near the entrance for return-
ing foragers. They take the crop load of returning for-
agers and head towards the storage area (see Figure 1).
They drop their nectar load there and head back towards
the entrance.

Nurse bees: These bees navigate (uphill) in the chemical
stimulus emitted by hungry larvae.
If they are located
on a patch containing a hungry larva, they start to feed
this larva until it is saturated or the nurse is almost empty.
These feedings last for several time steps.

Larvae: The brood resides in cells (patches) in the central
broodnest area (see ﬁgure 1). Larvae cannot move. If they
have low nectar reserves, they emit a chemical hunger sig-
nal. See below for more details.

Modelling the Stimuli
In our model it is important that stimuli differ signiﬁcantly
in their dynamics and in their range: Contact stimuli are
emitted by returning forager bees to attract storer bees to
take over the nectar load. These signals have a short range
(r = 1) only and stop immediately after the forager is en-
tirely unloaded. Depending on the waiting period a forager
searched for a storer bee, it then performs either a ’wag-
gle dance’ (Tsearch <= 20) to recruit more forager bees or
a ’tremble dance’ (Tsearch >= 50) to recruit more storer

Artificial Life XI 2008

 530

αi(t) = 1 −

vi(t)
crlow · capacitylarva

(2)

If the larva has more nectar in its reserves, then the value
of αi(t) is set to 0. A hungry larva at position x is re-
ferred as larva i. The nectar reserve of this larva is de-
scribed as vi(t), the maximum storing capacity of a larva
was set to capacitylarva = 0.33. The ’diffusion term’ was
implemented numerically (and discrete): we used the build-
in function ”diffuse” available in the NetLogo programming
environment. The light stimulus decreases linearly with in-
creasing distance from the hive’s entrance and is used for
navigation of foragers for leaving the hive and for naviga-
tion of storer bees for approaching the entrance area and for
approaching the honey area. Nurse bees navigate uphill in
the chemical pheromone ﬁeld to ﬁnd hungry larvae to feed
and move towards darker areas to ﬁnd honey cells for reﬁlls.

Simulated physiology
An adult bee can hold a maximum of 1 unit of crop
load. A larva can hold 0.33 units at maximum. Adult
bees consume their nectar loads at a low rate of crlow =
0.0004 units/step, ﬂying foragers consume at a higher rate
crhigh = 0.001 units/step. Larvae consume nectar at the
rate crlarva = 0.0004 units/step. If an agent (bee or larva)
runs out of nectar, it dies and is removed from the system.
The bottom of ﬁgure 2 shows these consumption ﬂows.

Modelling Division of Labour
The most important aspect in our model is the implementa-
tion of the task selection mechanism. We followed the ap-
proaches of Gautrais et al. (2002) and implemented a thresh-
old based system. Each type of local stimulus can motivate
an unemployed adult bee agent (task = ’no-task’) to join one
of the tasks m ∈ { ’foraging’, ’storing’, ’nursing’ }. See
ﬁgure 2 (upper part) for the possible task transitions. When-
ever one of these stimuli exceeds an individual threshold of
an agent i located on that patch x, the agent engages in the
associated task m. Each of these thresholds is modelled in a
non-linear manner, as is shown by equation 3. pi,m models
the likelihood to engage in task m in one time step. sx,m
is the local intensity of the task-associated stimulus. Θi,m
is used to shift the threshold individually up and down, n
is used to express the degree of non-linearity in these be-
havioural decisions.

pi,m =

sn
x,m
x,m + Θn
sn

i,m

(3)

Employed bees switch back to the unemployed state with
probabilities of λ′nursing′ = λ′storing′ = 0.005/step and
λ′f oraging′ = 0.001/step. To allow specialisation within
this system, the levels of the thresholds are adapted individ-
ually during run time. In the case that an unemployed agent
engages in task m′, the Θi,m is reduced by ξm, making it

Figure 2: The ﬂow of nectar, bees’ metabolism and task se-
lection of our agents. Top: Individual task selection depicted
as a state automaton. Middle: Most important regulation
feedbacks. Bottom: Task cohorts as compartments in the
ﬂow of nectar in the colony. Rounded boxes represent in-
dividual tasks. Solid arrows indicate task switches. Dashed
arrows indicate dependencies (’A is affecting B’). Rectangu-
lar boxes represent worker cohorts, larvae or combs. Solid
arrows indicate nectar ﬂows. The ﬂower represents a nectar
source, the cross-like symbols represent sinks.

bees. Both stimuli spread wider (r = 3) and decay non-
linearly ( 1
d ) with increasing distance from the emitting bee.
As soon as the dancing bee stops, the emitted dance sig-
nal disappears also from all other patches immediately. In
contrast to that, the chemical stimuli emitted by larvae stay
much longer and spread wider: They diffuse to all nearby
patches and decay slowly:

∂C(x)
∂t

= D∇2C(x) − µC(x) + αi(t)Lhungry(x),

(1)

where C(x)

is the local concentration of hunger
pheromone at position x, µ is the rate of pheromone decay,
α is the addition rate of pheromone produced by a hungry
larva. Lhungry(x) is set to 1 in case that there is a hungry
larva at position x, else it is set to 0.

In case that the larva at position x has a nectar reserve
below the hunger threshold crlow = 0.25, alpha scales linear
from 1 down to 0, as described in equation 2.

Artificial Life XI 2008

 531

more likely that the agent will engage in this task in future.
Whenever an unemployed agent does not engage in a task,
the corresponding threshold is increased by ϕm, making it
more unlikely that these behaviours will be triggered later
on. In our simulations, all values of ξ were set to ξ = 0.1
and all values of ϕ were set to ϕ = 0.001. It was shown
in (Schmickl and Crailsheim, 2008b) that these parameter
values lead to plausible division of labour. In our simula-
tions we used n = 2 for all agents and all agents initially
started with Θ values of 0.001 for all tasks. During run time,
values of Θ were conﬁned between 0 and 1.

Initial Conditions
Our simulations were conducted with 700 adult bee agents
and 100 larvae. The larvae were distributed randomly (nor-
mal distribution) around the center of the hive. All adult
agents started in randomized positions and with randomized
headings. Their initial task was set to ’no-task’. All agents
had (uniformly) randomized crop loads.

Results
This article focuses on the aspects associated with the reg-
ulation of brood nursing, thus we manipulated the ratio of
adult bees to larvae in our simulation experiments described
here. We ﬁrst simulated 10000 time steps of an undisturbed
colony, to allow the colony to reach equilibrium in brood
supply and in division of labour. At time step 10000, the
whole simulation state was saved on hard disk. Starting
with this saved conﬁgurations several perturbations were
performed (addition of brood, removal of adult workers) and
the resulting changes in task cohorts were measured.
In
these experiments, all adult bees started with Θ values of
0.

Perturbations of the adult-to-brood ratio

Figure 3: Removal of brood affects the size of the nursing
cohort strongly. The additional workforce that gets avail-
able from abandoning brood care affects also the size of the
other working cohorts. The arrows indicate the timing of the
perturbation. Graphs show mean values (N=6).

The more brood was removed at time step 10000, the less
bees performed the nursing task. This high abandonment
from nursing made more bees available for the tasks of stor-
ing and for the foraging task, as can be seen in ﬁgure 3.

Artificial Life XI 2008

 532

Figure 4: Addition of brood affects the size of the nursing
cohort strongly. This binds additional workforce to the task
of nursing, what in turn affects also the size of the foraging
cohort and of the storing cohort. The arrows indicate the
timing of the perturbation. Graphs show mean values (N=6).

Analogously we observed a signiﬁcant increase of the size
of the nursing cohort as we spontaneously added brood to
the colony at time step 10000. This reduced the number of
unemployed bees, in turn affecting also the task equilibrium
of foraging bees and of storing bees, as shown in ﬁgure 4.

Figure 5: Removal of worker bees affected all task cohorts.
The cohort of nurses bees was strongly affected only with
the more extreme removal of worker bees. The arrows in-
dicate the timing of the perturbation. Graphs show mean
values (N=6).

As ﬁgure 5 shows, the removal of adult bees strongly af-
fected all three task cohorts. The removal was a random
pick across all task cohorts. While the cohorts of foragers
and storers were affected signiﬁcantly by all worker losses,
the cohort of nurse bees was affected signiﬁcantly only by
the bigger losses of worker bees.

Nectar economics

During the experiments shown in the ﬁgures 3, 4, and 5,
the colony structure was signiﬁcantly altered at time step
t = 10000. Since we also modelled the ﬂow of nectar (nec-
tar income and consumption), we could also investigate how
these alteration affected the colony’s nectar economics. Fig-
ure 6 shows these results: The removal of brood strongly
enhanced the colony’s net nectar gain, as a signiﬁcant sink
for nectar was decreased. In contrast to that, the addition
of brood increased this important nectar sink, what had a
detrimental effect on the colony’s nectar economics. This
effect was also observed by leaving the sink unchanged but

by decreasing the foraging workforce, as it happened by the
removal of adult bees.

Figure 6: Removal of brood leads to strong increases of the
colony’s net nectar gain over time. Addition of brood or re-
moval of workers lead to strong decreases of the colony’s net
nectar gain. The arrows show the timw of the perturbation.
All graphs show mean values (N=6).

Scaling properties of division of labour
As shown above, colony manipulations affected the task co-
horts and the colony’s net nectar gain. As ﬁgure 7 shows,
forager and storer cohorts are severely affected (high steep-
ness of the regression curve) by removal of worker bees.
Nurses and net nectar gain is affected by both, brood ma-
nipulation and by adult removal. Almost all correlations be-
tween perturbation strength and resulting cohort sizes were
found to be linear.
It has to be mentioned that the steep-
ness of regressions varied signiﬁcantly, what points towards
different sensitivities of cohorts to perturbation types.

Nursing on the individual level
Empirical studies showed that nursing of brood is regulated
in a supply-demand driven process. Huang and Otis (1991a)
performed an interesting study, where they prevented a set

Figure 7: The effects of all perturbations scaled in most
cases linearly with the strengths of the perturbations. The
ﬁgures indicate the relative difference of the end result of
the simulations (t = 20000) compared to the undisturbed
control simulations. Graphs show mean values (N=6).

of 4-day old larvae from being fed by nurse bees with a cage
that was placed around the larvae. They found that these
starved larvae were fed preferentially by nurse bees after the
cage was removed. We were interested whether or not such
effects could be observed also in our model. Thus we (vir-
tually) put a cage around a central spot in the center of the
broodnest, what prevented the bees from entering this area.
After 400 time steps with the cage preventing feedings, a
fraction of the larvae died (see ﬁgure 8). The remaining lar-
vae were fed preferentially during the ﬁrst 1000 time steps
after the cage was removed (ﬁgure 9). Later on, the for-
merly starved larvae were fed on average on the same level,
but the mode of nursing was still altered due to experimental
manipulation: Feedings were performed in a more oscillat-
ing manner, suggesting that disturbances of brood nursing
could cause long-term alterations in the colonies nursing be-
haviour. Table 1 sums up the mean number of feedings per
time-slot (which was 100 time steps wide) per larva for both

Artificial Life XI 2008

 533

zones (central cage area, peripheral ’no-cage’ area):

Phase (time steps)

Pre (0 - 1500)
Experiment (1501 - 1900)
Post (1901 - 2900)
End (2901 - 5000)

Central area
(cage)
0.24 ± 0.16
0 ± 0
0.37 ± 0.64
0.24 ± 0.34

Peripheral area
(no-cage)
0.24 ± 0.13
0.24 ± 0.14
0.22 ± 0.11
0.23 ± 0.08

Table 1: Statistical comparison of all 4 phases in both exper-
imental zones. Means values were gained from all larvae in
the corresponding zone per 100 time steps. ± indicates the
corresponding standard deviations in these datasets.

Figure 9: (A) In the area outside the cage-zone, larvae are
fed in all experimental periods on the same average rate. (B)
In the pre-experimental period, central and peripheral larvae
are fed on the same level. During the cage-period, no feed-
ings can occur. During the ﬁrst 1000 time steps after the
cage’s removal, the remaining starved larvae are fed prefer-
entially. Also the oscillations increased signiﬁcantly. In the
ﬁnal period, the level of feedings returns to the initial value,
but the rhythmicity stays on an increased level.

randomized uniformly between 0 and 1. After 10000 time
steps, the values of Θnursing were measured. To speed up
the specialization process, all ξm values were increased to
0.2 in these experiments.

The ﬁrst simulation we performed was with the colony
status we used also in the simulations described in the previ-
ous sections (700 adults, 100 larvae). As can be seen in ﬁg-
ure 10, approx. 10% of the bees developed into highly spe-
cialized nurses. The majority of bees developed into highly
specialized storage bees or into ’partly-specialized’ storage
bees. Although we observed between 65 and 120 forager
bees throughout the run-time of the simulation, the observed
degree of specialization for this task was not comparable to

Figure 8: (A) Initially the brood starts hungry and emits a
lot of hunger signals. (B) After 300 time steps, the nurses
satisﬁed the brood and kept it on a rather fed status. (C) At
time step t = 1500, the (virtual) cage was installed around
the central brood nest area. At t = 1600, many hungry
larvae can be found in the center, emitting strong hunger
stimuli.
(D) After removal of the cage at t = 1900, the
central brood is either dead (removed) or very hungry. At
t = 2000, nurses aggregate in this area and feed frequently.

Specialization in the nursing task

In additional simulation runs, the development of thresholds
in big and small colonies with low and high brood state was
observed. In these experiments, all Θ values were initially

Artificial Life XI 2008

 534

the degree of specialization of the other tasks. In (Gautrais
et al., 2002) it is predicted by another model, that the de-
gree of specialization increases with colony size. To investi-
gate this question also with our model, we scaled down the
colony size (adults and brood) by the factor 1
4 . Please note
that the nursing workload per bee was kept constant. As can
be seen in ﬁgure 11, the degree of specialization decreased,
especially with the nursing task and with the storing task.

Figure 10: Degree of specialization to our modelled tasks
in a simulated colony consisting of 700 adult bees and 100
larvae. Low theta values (Θ <= 0.2) are interpreted as ’high
degree’ of specialization. Values of 0.2 < Θ < 0.8 are
interpreted as partially specialized bees. Higher values of Θ
are interpreted as bees not specialized to the speciﬁc task.

Figure 11: Degree of specialization to our modelled tasks
in a simulated colony consisting of 175 adult bees and 25
larvae. Low theta values (Θ <= 0.2) are interpreted as ’high
degree’ of specialization. Values of 0.2 < Θ < 0.8 are
interpreted as partially specialized bees. Higher values of Θ
are interpreted as bees not specialized to the speciﬁc task.

In a ﬁnal simulation experiment, we doubled the work
load per adult bee compared to the settings shown in ﬁgure
10. As can be seen in ﬁgure 12, this increased preferentially
the degree of specialization of nurse bees, as the number of
highly-specialized nurses more than doubled.

Discussion
We showed that a threshold-based model can sufﬁce to sim-
ulate honeybee-speciﬁc division of labour. By performing
and analyzing our simulations, we learned that having a re-
active system of task cohorts, which is able to react plausi-
bly to perturbations of colony structure, is not a guarantee

Artificial Life XI 2008

 535

Figure 12: Degree of specialization to our modelled tasks
in a simulated colony consisting of 700 adult bees and 200
larvae. Low theta values (Θ <= 0.2) are interpreted as ’high
degree’ of specialization. Values of 0.2 < Θ < 0.8 are
interpreted as partially specialized bees. Higher values of Θ
are interpreted as bees not specialized to the speciﬁc task.

for having task specialisation and de-specialization of work-
ers: As our ﬁgures 3, 4, 5 and 6 show, our modelled colony
reacts very plausibly to the induced perturbations. When we
investigated whether or not the nursing received by individ-
ual larvae after a deprivation experiment is predicted plausi-
bly by our multi-agent simulation, we found that the nursing
regulation reﬂects empiric results very well.

Although we found division of labour and specialization
of hive-bees (nurses, storers), ﬁgure 10 tells us, that the for-
aging cohort did not show the expected high degree of spe-
cialisation in our simulations. The threshold-response sys-
tem was able to model specialisation of nurses by chemical
brood stimuli at a very high degree. We found also many
bees highly specialized to the task of storing. But also a
high number of ’semi-specialized’ storers (0.2 < Θ < 0.8)
was found. Most foragers showed only a low degree of spe-
cialization, indicating that foragers are not often re-recruited
to the foraging task after they abandoned from foraging.
We had between 65 and 120 foraging bees present at all
times, but almost all foraging bees performed just one or
two consecutive engagements, having several round-trips.
The fact that we still observed task cohorts that reacted
adaptively to perturbations can be reasoned by the equi-
libria that emerge: Foraging has a high turn-over number,
that means foragers that quit the task once in our model are
not often re-recruited. But simultaneously many other bees,
that performed other tasks before, are recruited to the task
of foraging. Obviously, this sufﬁces to allow an adaptive
equilibrium-based division of labour.

We conclude that

threshold reinforcement (Theraulaz
et al., 1998; Gautrais et al., 2002) is well suited to pro-
duce plausible specialization in tasks that are associated with
very durable, time-persistent and spreading stimuli, like the
pheromones that stimulate nursing behavior. Also the stor-
ing task has a high density of stimuli (tremble dances, every
returning forager emits the ’storing’ stimulus), but forag-
ing is induced only by the (relatively rare) waggle dances.

As these dances do not occur at a comparably high fre-
quency (only some foragers perform a waggle dance), re-
recruitment is hard to explain by just this stimuli alone. In
nature, foragers are a well specialized group in the honeybee
society, thus we can assume that we will have to incorporate
other additional factors to achieve the observed high spe-
cialisation of forager bees: shaking signals, stop signals and
a (probably age-related) higher predisposition for the for-
aging task, as it can be easily implemented into our model
by a slight downward-bias of Θf oraging in a speciﬁc group
of bees which represent ’older’ bees. In addition, motiva-
tion for foraging can be also inﬂuenced by physiological
properties of the bees, which reﬂect often characteristic hive
conditions, as was demonstrated in (Camazine, 1993). We
conclude that, incorporating additional regulatory systems
and motivational aspects of bees, as for example temporal
polyethism (Seeley, 1982; Johnson, 2005) can signiﬁcantly
improve the models predictions concerning foraging special-
ization.

For interpreting the observed differences in task special-
ization, we had to consider also the regions of the hive in
which the recruited workers tend to stay. These regions in-
clude a speciﬁc mixture of stimuli, thus determining also
the likelihoods to switch to other tasks. Such effects are dis-
cussed in the ’foraging-for-work’-theory, as it is described
in (Franks and Tofts, 1994). By working with and on our
model we learned that honeybee-speciﬁc division of labour
cannot be modelled with the threshold-reinforcement model
alone. We developed the idea that several of the discussed
concepts of honeybees’ division of labour have to be im-
plemented into one single model, which then represents an
integrative approach to understand honeybee’s’ division of
labour. By extending our model in these directions, we will
pursue this scientiﬁc goal.

Acknowledgements
This work was supported by: EU-IST FET project ’I-
Swarm’, no. 507006; EU-IST-FET project ’SYMBRION’,
no. 216342; EU-ICT project ”REPLICATOR”, no. 216240.
Austrian Science Fund (FWF) research grants: P15961-B06
and P19478-B16.

References
Anderson, C. (1998). Simulation of the feedbacks and regulation
of recruitment dancing in honey bees. Advances in Complex
Systems, 1:267–282.

Beshers, S. N. and Fewell, J. H. (2001). Models of division of labor
in social insects. Annual Review of Entomology, 46:413–440.

Beshers, S. N., Huang, Z. Y., Oono, Y., and Robinson, G. E. (2001).
Social inhibition and the regulation of temporal polyethism in
honey bees. Journal of Theoretical Biology, 213:461–479.

Camazine, S. (1993). The regulation of pollen foraging by honey
bees: how foragers assess the colony’s need for pollen. Be-
havioral Ecology and Sociobiology, 32:265272.

Artificial Life XI 2008

 536

Franks, N. R. and Tofts, C. (1994). Foraging for work: How tasks

allocate workers. Animal Behaviour, 48:470–472.

Gautrais, J., Theraulaz, G., Deneubourg, J.-L., and Anderson, C.
(2002). Emergent polyethism as a consequence of increased
colony size in insect societies. Journal of Theoretical Biol-
ogy, 215:367–373.

Huang, Z.-Y. and Otis, G. W. (1991a). Inspection and feeding of
larvae by worker honey bees (hymenoptera: Apidae): effect
of starvation and food quantity. Journal of Insect Behavior,
4(2):305317.

Huang, Z.-Y. and Otis, G. W. (1991b). Nonrandom visitation of
brood cells by worker honey bees (hymanoptera: Apidae).
Journal of Insect Behavior, 4(2):177–184.

Johnson, B. R. (2005). Limited ﬂexibility in the temporal caste sys-
tem of the honey bee. Behavioral Ecology and Sociobiology,
58:219–226.

Lindauer, M. (1952). Ein Beitrag zur Frage der Arbeitsteilung
im Bienenstaat. Zeitschrift f¨ur vergleichende Physiologie,
34:299–345.

R¨osch, G. A. (1952). Untersuchungen ¨uber die Arbeitsteilung
im Bienenstaat. Zeitschrift f¨ur vergleichende Physiologie,
2:571–631.

Sakagami, S. F. (1953). Untersuchungen ¨uber die Arbeitsteilung in
einem Zwergvolk der Honigbiene. Beitr¨age zur Biologie des
Bienenvolkes, Apis mellifera L. Japanese Journal of Zoology,
2:117–185.

Schmickl, T. and Crailsheim, K. (2002). How honeybees (apis mel-
lifera l.) change their broodcare behaviour in response to non-
foraging conditions and poor pollen conditions. Behavioral
Ecology and Sociobiology, 51:415–425.

Schmickl, T. and Crailsheim, K. (2008a). An individual-based
model of task selection in honeybees. In et al., M. A., edi-
tor, Proceedings of the 10th International Conference on the
Simulation of Adaptive Behavior (SAB’08), Lecture Notes in
Artiﬁcial Intelligence 5040, pages 383–392, Springer-Verlag
Berlin, Heidelberg.

Schmickl, T. and Crailsheim, K. (2008b). Taskselsim: A model
of the self-organisation of the division of labour in honey-
bees. Mathematical and Computer Modelling of Dynamical
Systems, 14(2):101–125.

Seeley, T. D. (1982). Adaptive signiﬁcance of the age polyethism
schedule in honeybee colonies. Behavioural Ecology and So-
ciobiology, 11:287–293.

Seeley, T. D. (1992). The tremble dance of the honey bee: mes-
sage and meanings. Behavioral Ecology and Sociobiology,
31:375–383.

Theraulaz, G., Bonabeau, E., and Deneubourg, J.-L. (1998). Re-
sponse threshold reinforcement and division of labour in in-
sect societies. Proceedings of the Royal Society of London B,
265:327–332.

Wilensky,

U.

(1999).

NetLogo.
http://ccl.northwestern.edu/netlogo/. Center for Connected
Learning and Computer-Based Modeling, Northwestern
University. Evanston, IL.


