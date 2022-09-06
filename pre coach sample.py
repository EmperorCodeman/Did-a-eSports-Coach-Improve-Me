import math 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as st


# pre coach sample. 
# Uncomment/comment the pre post sections so they dont override each other and put in your scores to use, then comment out and swap to post coach sample if you want graph
game_names = np.array(['Game 2', 'Game 3', 'Game 4', 'Game 5', 'Game 6'])
accuracy = [.30, .25, .22]
accuracy_readable = ["30%", "25%", "22%"]
eliminations = [0,0,0,0,4]
damage_to_players = [0,0,323,90,679]
headshots = [1,0,1]
damage_to_self = [128, 194, 228, 146, 347]
#pre hypothesis testing prep
accuary_avg_pre = np.average(accuracy)
accuracy_std_pre = np.std(accuracy)
eliminations_avg_pre = np.average(eliminations)
eliminations_std_pre = np.std(eliminations)
damage_to_players_avg_pre = np.average(damage_to_players)
damage_to_players_std_pre = np.std(damage_to_players)
headshots_avg_pre = np.average(headshots)
headshots_std_pre = np.std(headshots)
damage_to_self_avg_pre = np.average(damage_to_self)
damage_to_self_std_pre = np.std(damage_to_self)


# post coachy sample 
game_names = np.array(['Game 1', 'Game 2', 'Game 3', 'Game 4', 'Game 5', 'Game 6', 'Game 7', 'Game 8', 'Game 9', 'Game 10'])
accuracy =          [.25, .15, .11, .29, .2,  .16, .22, .18, .28, .13]
eliminations =      [1,0, 1,   3,   1,   1,   1,   1,   3,   1,   1] 
damage_to_players = [158, 69,  259, 353, 160, 207, 191, 521, 277, 258]
headshots =         [7,   0,   1,   2,   0,   0,   1,   0,   0,   1]
damage_to_self =    [206, 156, 243, 199, 178, 340, 216, 330, 187, 222]
accuracy_readable = ["25%", "15%", "11%", "29%", "20%", "16%", "22%", "18%", "28%", "13%"]
accuary_avg_post = np.average(accuracy)
eliminations_avg_post = np.average(eliminations)
damage_to_players_avg_post = np.average(damage_to_players)
headshots_avg_post = np.average(headshots)
damage_to_self_avg_post = np.average(damage_to_self)

#GRAPHS of the vars
#accuracy. If no shots no accuracy, throw out 0's
plot1 = plt.figure(1)
plt.style.use('ggplot')
x_pos = [i for i, _ in enumerate(accuracy)]
plt.axhline(y=np.average(accuracy), color='b', linestyle='--', label='Average Accuracy')
plt.legend()
plt.bar(x_pos, accuracy, color='blue')
plt.xlabel("Game Number")
plt.ylabel("Accuracy")
plt.title("Firearm Accuracy: Hits / (Shots Fired)")
plt.xticks(x_pos, game_names)
plt.yticks(accuracy, accuracy_readable)
plt.plot()

#elimination
plot1 = plt.figure(2)
plt.style.use('ggplot')
x_pos = [i for i, _ in enumerate(eliminations)]
plt.axhline(y=np.average(eliminations), color='b', linestyle='--', label='Average Eliminations')
plt.legend()
plt.bar(x_pos, eliminations, color='green')
plt.xlabel("Game Number")
plt.ylabel("Kills")
plt.title("Enemy Body Count: Deaths I Inclifted per Match")
plt.xticks(x_pos, game_names)
plt.plot()

#damage to players
plot1 = plt.figure(3)
plt.style.use('ggplot')
x_pos = [i for i, _ in enumerate(damage_to_players)]
plt.axhline(y=np.average(damage_to_players), color='b', linestyle='--', label='Average Damage Inflicted')
plt.legend()
plt.bar(x_pos, damage_to_players, color='green')
plt.xlabel("Game Number")
plt.ylabel("Damage I Inflicted on Enemy")
plt.title("Damage I Inflicted: Enemy Hit points lost per Match")
plt.xticks(x_pos, game_names)
plt.plot()

#damage incured
plot1 = plt.figure(4)
plt.style.use('ggplot')
x_pos = [i for i, _ in enumerate(damage_to_self)]
plt.axhline(y=np.average(damage_to_self), color='b', linestyle='--', label='Average Damage Sustained')
plt.legend()
plt.bar(x_pos, damage_to_self, color='red')
plt.xlabel("Game Number")
plt.ylabel("Damage Enemy Inflicted on me")
plt.title("Damage Inflicted to Self: My Hit points lost per Match")
plt.xticks(x_pos, game_names)
plt.plot()

#damage to headshots
plot1 = plt.figure(5)
plt.style.use('ggplot')
x_pos = [i for i, _ in enumerate(headshots)]
plt.axhline(y=np.average(headshots), color='b', linestyle='--', label='Average Head Shots')
plt.legend()
plt.bar(x_pos, headshots, color='green')
plt.xlabel("Game Number")
plt.ylabel("Head Shots")
plt.title("Head Shots: Head Shots I scored / Match ")
plt.xticks(x_pos, game_names)
plt.plot()

# Hypothesis Testing 
# z score: is devation from the average in terms of standard deviation, in this case we are treating the new average as if its from the same distribution in order to determine the probability that the new value is a within the distribution and not from a new distrabution
z_elimations = (eliminations_avg_post - eliminations_avg_pre) / eliminations_std_pre 
z_accuracy = (accuary_avg_post - accuary_avg_pre) / accuracy_std_pre
z_headshots = (headshots_avg_post - headshots_avg_pre) / headshots_std_pre
z_damage_to_self = (damage_to_self_avg_post - damage_to_self_avg_pre) / damage_to_self_std_pre
z_damage_to_players = (damage_to_players_avg_post - damage_to_players_avg_pre) / damage_to_players_std_pre
z_scores = np.array([z_elimations, z_accuracy, z_headshots, z_damage_to_self, z_damage_to_players])

# Probability that a change DID occure and change is NOT due to normal variance , range 0-1. 
# Note normal distribution is symetric, thus I use abs to reflect negative z's to positive since they process the same in this schema  
# probablity is the integral of the cdf with limits -z to z 
probabilites_change_occured = np.array([2*st.norm.cdf(z) - 1 for z in np.abs(z_scores)]) # https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa

# Unit for all changes are the same. Unit is percentage change. 
# Note unit / unit cancels the unit thus all variables develope the same unit and become legal operands to arithmitic operators in the operations to follow in avg
# TODO rework all to inductive matrix changes = np.array([for ])
# Note: these are not percentages stricktly. You need to multiply * 100 for that exp. .2 => .2*100 = 20%
# Note: That no change is 0 and bad change is negative. Improvement is postive change
eliminations_change = eliminations_avg_post/eliminations_avg_pre - 1
accuracy_change = accuary_avg_post/accuary_avg_pre - 1
headshots_change = headshots_avg_post/headshots_avg_pre - 1
damage_to_self_change = damage_to_self_avg_post/damage_to_self_avg_pre - 1
damage_to_players_change = damage_to_players_avg_post/damage_to_players_avg_pre - 1
changes = np.array([eliminations_change, accuracy_change, headshots_change, damage_to_self_change, damage_to_players_change])

# Signed Weights allow us to assign significance to variables, good is positive, bad is negative. These weights are subjective 
eliminations_weight = 3
accuracy_weight = .5
headshots_weight = .5
damage_self_weight = -.5  # Note that if you negativly change a negative var then you have improved. This var is not worse performance. The weight interacts with the change to result in performance of negative or positive  
damage_others_weight = 1
weights = np.array([eliminations_weight, accuracy_weight, headshots_weight, damage_self_weight, damage_others_weight])
#net_weight = np.sum(weights) # note. Never use net weight. Ponder, if you avgerage 3 numbers-> 1*2, -1*3, 1*5 you divide my three, not the net of the weights which is 1 + -1 + 1 = 1 
absolute_weight = np.sum(np.abs(weights))
# integrate to singular conclusion using weighted averages 
#Note that negative z scores integrate with positive thus a net weighted z occures with positive as improvement 
net_avg_z = np.sum(weights*z_scores) / absolute_weight # positive is good change % and vice versa #(z_elimations*eliminations_weight + z_accuracy*accuracy_weight + z_headshots*headshots_weight + z_damage_to_self*damage_self_weight + z_damage_to_players*damage_others_weight) / sum_weights
absolute_avg_z = np.sum(np.abs(weights*z_scores)) / absolute_weight # indicator of non trivial general change in terms of z score of normal distribution
probabilty_improved_avg = 2*st.norm.cdf(abs(net_avg_z)) - 1  # Probability of improvement alligned with probablity axioms ie. 0-1.0 probablity
probabilty_any_change_avg = 2*st.norm.cdf(abs(absolute_avg_z)) - 1   
# change is in units of percentage. Net effect is to value the significance of the variable 
improved_avg = np.sum(weights*changes) / absolute_weight # positive is good change and vice versa
change_avg = np.sum(np.abs(weights*changes)) / absolute_weight

# Now finally we multiply times the probablity to render the final actionable insight
# Review: Weights are signed, changes are signed, and probs and absolute_weight are unsigned. A negative change on a negative weight is an improvement 
nontrivial_changes =        weights*changes*probabilites_change_occured / absolute_weight
nontrivial_change =         np.sum(np.abs(nontrivial_changes)) # it seems weird that the avg is higher than each vars nontrivial value but this is correct, because its a weighted average
nontrivial_improvement =    np.sum(nontrivial_changes) 
nontrivial_percent_change = str(nontrivial_change*100) + "%"
nontrivial_percent_improvement = str(nontrivial_improvement*100) + "%"


# Graphs of Conclusion

#Probablity of change graph
plot1 = plt.figure()
plt.style.use('ggplot')
x = ['elimations', 'accuracy', 'headshots', 'damage_to_self', 'damage_to_players']
x_pos = [i for i, _ in enumerate(x)]
plt.axhline(y=np.sum(probabilites_change_occured*weights/absolute_weight)*100, color='r', linestyle='--', label='Weighted Avg Probablity')
plt.legend()
plt.bar(x_pos, probabilites_change_occured*100, color='green')
plt.xlabel("Variable Names")
plt.ylabel("Integral of mdf for normal distribution: z score of X bar dis")
plt.title("% Probability variable changed from using Coach")
plt.xticks(x_pos, x)
#plt.show()
plt.plot()

#Change occured graph
plot1 = plt.figure()
plt.style.use('ggplot')
x = ['elimations', 'accuracy', 'headshots', 'damage_to_self', 'damage_to_players']
x_pos = [i for i, _ in enumerate(x)]
plt.axhline(y=change_avg*100, color='r', linestyle='--', label='Weighted avg Change')
plt.legend()
plt.bar(x_pos, changes*100, color='green')
plt.xlabel("Variable Names")
plt.ylabel("% Change ocured between pre/post coach, positive is improvment")
plt.title("% How variables changed")
plt.xticks(x_pos, x)
#plt.show()
plt.plot()


# Nontrivail change graph
plot1 = plt.figure()
plt.style.use('ggplot')
x = ['elimations', 'accuracy', 'headshots', 'damage_to_self', 'damage_to_players']
x_pos = [i for i, _ in enumerate(x)]
plt.axhline(y=nontrivial_change*100, color='r', linestyle='--', label='Importance X Probability X Change -> Avg')
plt.legend()
plt.bar(x_pos, nontrivial_changes*100, color='green')
plt.xlabel("Variable Names")
plt.ylabel("% Importance of Variable * Probablity change is not fluke * Pre/Post Change")
plt.title("% Non Trivial Change of Variables")
plt.xticks(x_pos, x)
plt.show()
plt.plot()



