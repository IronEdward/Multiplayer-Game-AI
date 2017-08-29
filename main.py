from pygame.locals import *
import sys
from time import sleep
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
-----Some Specifications-----

The opponent of the agent(Red) is a simple algorithm that gets stronger over episodes. See code for details.

Bot(Blue):                               Random Mode -> Medium Mode -> Boss Mode
Model:                                   Q Learning with experience replay and a target network
Layers:                                  2
Experience Replay:                       Yes
Players:                                 Square-Shaped
Parameters:                              10
Actions:                                 9
Bullet:                                  Beam
Activation Function:                     Linear
Target Network:                          Yes
"""



#Screen Setup
disp_x, disp_y = 1000, 800
arena_x, arena_y = 600, 600
border = 4; border_2 = 1

#Color Setup
white = (255, 255, 255); aqua= (0, 200, 200)
red = (255, 0, 0); green = (0, 255, 0)
blue = (0, 0, 255); black = (0, 0, 0)
green_yellow = (173, 255, 47); energy_blue = (125, 249, 255)

#Initialize character positions
init_character_a_state = [disp_x/2 - arena_x/2 + 50, disp_y/2 - arena_y/2 + 50]
init_character_b_state = [disp_x/2 + arena_x/2 - 50, disp_y/2 + arena_y/2 - 50]

#Setup character dimentions
character_size = 50
character_move_speed = 25

#Initialize character stats
character_init_health = 100

#initialize bullet stats
beam_damage = 10
beam_width = 10
beam_ob = -100
beam_cooldown_time = 10

#The Neural Network
input_layer = tf.placeholder(shape=[1,10],dtype=tf.float32)
weight_1 = tf.Variable(tf.random_uniform([10,9],0,0.01))

#The calculations, loss function and the update model of the NN
Q = tf.matmul(input_layer, weight_1)
predict = tf.argmax(Q, 1)
next_Q = tf.placeholder(shape=[1,9],dtype=tf.float32)

loss = tf.reduce_sum(tf.square(next_Q - Q))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
updateModel = trainer.minimize(loss)

#The Target Network
target_input_layer = tf.placeholder(shape=[1,10],dtype=tf.float32)
target_weight_1 = tf.Variable(tf.random_uniform([10,9],0,0.01))

#The calculations, loss function and the update model of the Target NN
target_Q = tf.matmul(target_input_layer, target_weight_1)
target_next_Q = tf.placeholder(shape=[1,9],dtype=tf.float32)

initialize = tf.global_variables_initializer()

jList = []
rList = []

init()
font.init()
myfont = font.SysFont('Comic Sans MS', 15)
myfont2 = font.SysFont('Comic Sans MS', 150)
myfont3 = font.SysFont('Gothic', 30)
disp = display.set_mode((disp_x, disp_y), 0, 32)

def calculate_score(total_reward, turns, win_or_lost):
    return (win_or_lost * 1000 + total_reward * 0.5 + 10000./turns * win_or_lost)/17.


#CHARACTER/BULLET PARAMETERS
agent_x = agent_y = int()
bot_x = bot_y = int()
agent_hp = bot_hp = int()
bot_beam_dir = int()
agent_beam_fire = bot_beam_fire = bool()
agent_beam_x = bot_beam_x = agent_beam_y = bot_beam_y = int()
agent_beam_size_x = agent_beam_size_y = bot_beam_size_x = bot_beam_size_y = int()
agent_beam_cooldown = bot_beam_cooldown = int()
agent_beam_cooldown_active = bot_beam_cooldown_active = bool()
bot_current_action = agent_current_action = int()

#MISC PARAMETERS
epoch = int()

def param_init():
    """Initializes parameters"""
    global agent_x, agent_y, bot_x, bot_y, agent_hp, bot_hp, agent_beam_fire, bot_beam_fire, agent_beam_x, bot_beam_x, agent_beam_y, bot_beam_y

    agent_x = list(init_character_a_state)[0]; agent_y = list(init_character_a_state)[1]
    bot_x = list(init_character_b_state)[0]; bot_y = list(init_character_b_state)[1]
    agent_hp = bot_hp = character_init_health
    agent_beam_fire = bot_beam_fire = False
    agent_beam_x = bot_beam_x = agent_beam_y = bot_beam_y = beam_ob
    agent_beam_size_x = agent_beam_size_y = bot_beam_size_x = bot_beam_size_y = 0
    agent_beam_cooldown = bot_beam_cooldown = 0
    agent_beam_cooldown_active = bot_beam_cooldown_active = False



def screen_blit():
    """Prints shapes to the display"""
    global disp, disp_x, disp_y, arena_x, arena_y, border, border_2, character_size, agent_x, \
    agent_y, bot_x, bot_y, character_init_health, agent_hp, bot_hp, red, blue, aqua, green, black, green_yellow, energy_blue, \
    agent_beam_fire, bot_beam_fire, agent_beam_x, agent_beam_y, bot_beam_x, bot_beam_y, agent_beam_size_x, agent_beam_size_y, \
    bot_beam_size_x, bot_beam_size_y, beam_width, agent_beam_cooldown, bot_beam_cooldown

    agent_beam_cooldown_disp = myfont.render(('Cooldown: %d' % agent_beam_cooldown), 1, white)
    bot_beam_cooldown_disp = myfont.render(('Cooldown: %d' % bot_beam_cooldown), 1, white)
    disp.fill(black)
    draw.rect(disp, white, (disp_x / 2 - arena_x / 2 - border, disp_y /
                            2 - arena_y / 2 - border, arena_x + border * 2, arena_y + border * 2))
    draw.rect(disp, black, (disp_x / 2 - arena_x / 2,
                            disp_y / 2 - arena_y / 2, arena_x, arena_y))
    if bot_beam_fire == True:
        draw.rect(disp, green_yellow, (agent_beam_x, agent_beam_y, agent_beam_size_x, agent_beam_size_y))
        bot_beam_fire = False
    if agent_beam_fire == True:
        draw.rect(disp, energy_blue, (bot_beam_x, bot_beam_y, bot_beam_size_x, bot_beam_size_y))
        agent_beam_fire = False

    draw.rect(disp, red, (agent_x, agent_y, character_size, character_size))
    draw.rect(disp, blue, (bot_x, bot_y, character_size, character_size))

    draw.rect(disp, red, (disp_x / 2 - arena_x / 2 - border, disp_y / 2 + arena_y / 2 +
                            border + border_2, float(agent_hp) / float(character_init_health) * 100, 10))
    draw.rect(disp, blue, (disp_x / 2 + arena_x / 2 + border - 100, disp_y / 2 + arena_y / 2 +
                            border + border_2, float(bot_hp) / float(character_init_health) * 100, 10))
    disp.blit(agent_beam_cooldown_disp, (disp_x / 2 - arena_x / 2 - border, disp_y / 2 + arena_y / 2 +
                            border * 2 + 10 + border_2))
    disp.blit(bot_beam_cooldown_disp, (disp_x / 2 + arena_x / 2 + border - 100, disp_y / 2 + arena_y / 2 +
                            border * 2 + 10 + border_2))



def bot_take_action():
    """Function that returns the actions of the bot"""
    global epoch, agent_x, agent_y, bot_x, bot_y, character_size
    if epoch < 200:
        if random.randint(1, 100) > 95:
            return random.randint(1, 4)
        else:
            return random.randint(5, 9)
    elif 200 <= epoch < 2000:
        if agent_y < bot_y + character_size/2 <= agent_y + character_size:
            if random.randint(0, 100) > 90:
                if agent_x <= bot_x:
                    return 4
                else:
                    return 2
            else:
                return 9
        else:
            y_dist = abs(bot_y - agent_y)
            if random.randint(1, 100) > 5:
                if bot_y - agent_y <= 0:
                    return 7
                else:
                    return 5
            else:
                return random.randint(5, 9)
    elif 2100 <= epoch < 4000:
        if agent_y < bot_y + character_size/2 <= agent_y + character_size:
            if random.randint(0, 100) > 40:
                if agent_x <= bot_x:
                    return 4
                else:
                    return 2
            else:
                return 9
        else:
            y_dist = abs(bot_y - agent_y)
            if random.randint(1, 100) > 5:
                if bot_y - agent_y <= 0:
                    return 7
                else:
                    return 5
            else:
                return random.randint(5, 9)
    else:
        if agent_x < bot_x + character_size/2 < agent_x + character_size:
            if random.randint(0, 100) > 30:
                if agent_y <= bot_y:
                    return 1
                else:
                    return 3
            else:
                return 9
        elif agent_y < bot_y + character_size/2 <= agent_y + character_size:
            if random.randint(0, 100) > 30:
                if agent_x <= bot_x:
                    return 4
                else:
                    return 2
            else:
                return 9
        else:
            if random.randint(0, 100) > 30:
                x_dist = abs(bot_x - agent_x); y_dist = abs(bot_y - agent_y)
                if x_dist >= y_dist:
                    if bot_x - agent_x <= 0:
                        return 6
                    else:
                        return 8
                else:
                    if bot_y - agent_y <= 0:
                        return 7
                    else:
                        return 5
            else:
                return random.randint(1, 9)

def beam_hit_detector(player):
    global agent_x, agent_y, bot_x, bot_y, agent_beam_fire, bot_beam_fire, agent_beam_x, \
    bot_beam_x, agent_beam_y, bot_beam_y, agent_beam_size_x, agent_beam_size_y, \
    bot_beam_size_x, bot_beam_size_y, bot_current_action, agent_current_action, beam_width, character_size

    if player == "bot":
        if bot_current_action == 1:
            if disp_y/2 - arena_y/2 <= agent_y <= bot_y and (agent_x < bot_beam_x + beam_width < agent_x + character_size or agent_x < bot_beam_x < agent_x + character_size):
                return True
            else:
                return False
        elif bot_current_action == 2:
            if bot_x <= agent_x <= disp_x/2 + arena_x/2 and (agent_y < bot_beam_y + beam_width < agent_y + character_size or agent_y < bot_beam_y < agent_y + character_size):
                return True
            else:
                return False
        elif bot_current_action == 3:
            if bot_y <= agent_y <= disp_y/2 + arena_y/2 and (agent_x < bot_beam_x + beam_width < agent_x + character_size or agent_x < bot_beam_x < agent_x + character_size):
                return True
            else:
                return False
        elif bot_current_action == 4:
            if disp_x/2 - arena_x/2 <= agent_x <= bot_x and (agent_y < bot_beam_y + beam_width < agent_y + character_size or agent_y < bot_beam_y < agent_y + character_size):
                return True
            else:
                return False
    else:
        if agent_current_action == 1:
            if disp_y/2 - arena_y/2 <= bot_y <= agent_y and (bot_x < agent_beam_x + beam_width < bot_x + character_size or bot_x < agent_beam_x < bot_x + character_size):
                return True
            else:
                return False
        elif agent_current_action == 2:
            if agent_x <= bot_x <= disp_x/2 + arena_x/2 and (bot_y < agent_beam_y + beam_width < bot_y + character_size or bot_y < agent_beam_y < bot_y + character_size):
                return True
            else:
                return False
        elif agent_current_action == 3:
            if agent_y <= bot_y <= disp_y/2 + arena_y/2 and (bot_x < agent_beam_x + beam_width < bot_x + character_size or bot_x < agent_beam_x < bot_x + character_size):
                return True
            else:
                return False
        elif bot_current_action == 4:
            if disp_x/2 - arena_x/2 <= bot_x <= agent_x and (bot_y < agent_beam_y + beam_width < bot_y + character_size or bot_y < agent_beam_y < bot_y + character_size):
                return True
            else:
                return False

def action(agent_action, bot_action):
    global agent_x, agent_y, bot_x, bot_y, agent_hp, bot_hp, agent_beam_fire, \
    bot_beam_fire, agent_beam_x, bot_beam_x, agent_beam_y, bot_beam_y, agent_beam_size_x, \
    agent_beam_size_y, bot_beam_size_x, bot_beam_size_y, beam_width, agent_current_action, \
    bot_current_action, character_size, agent_beam_cooldown, agent_beam_cooldown_active, \
    bot_beam_cooldown, bot_beam_cooldown_active, beam_cooldown_time

    agent_current_action = agent_action; bot_current_action = bot_action
    reward = 0; cont = True; successful = False; winner = ""
    if 1 <= bot_action <= 4 and bot_beam_cooldown_active == False:
        bot_beam_fire = True
        bot_beam_cooldown_active = True
        bot_beam_cooldown = beam_cooldown_time
        if bot_action == 1:
            bot_beam_x = bot_x + character_size/2 - beam_width/2; bot_beam_y = disp_y/2 - arena_y/2
            bot_beam_size_x = beam_width; bot_beam_size_y = bot_y - disp_y/2 + arena_y/2
        elif bot_action == 2:
            bot_beam_x = bot_x + character_size; bot_beam_y = bot_y + character_size/2 - beam_width/2
            bot_beam_size_x = disp_x/2 + arena_x/2 - bot_x - character_size; bot_beam_size_y = beam_width
        elif bot_action == 3:
            bot_beam_x = bot_x + character_size/2 - beam_width/2; bot_beam_y = bot_y + character_size
            bot_beam_size_x = beam_width; bot_beam_size_y = disp_y/2 + arena_y/2 - bot_y - character_size
        elif bot_action == 4:
            bot_beam_x = disp_x/2 - arena_x/2; bot_beam_y = bot_y + character_size/2 - beam_width/2
            bot_beam_size_x = bot_x - disp_x/2 + arena_x/2; bot_beam_size_y = beam_width
    
    elif 5 <= bot_action <= 8:
        if bot_action == 5:
            bot_y -= character_move_speed
            if bot_y <= disp_y/2 - arena_y/2:
                bot_y = disp_y/2 - arena_y/2
            elif agent_y <= bot_y <= agent_y + character_size:
                bot_y = agent_y + character_size
        elif bot_action == 6:
            bot_x += character_move_speed
            if bot_x >= disp_x/2 + arena_x/2 - character_size:
                bot_x = disp_x/2 + arena_x/2 - character_size
            elif agent_x <= bot_x + character_size <= agent_x + character_size:
                bot_x = agent_x - character_size
        elif bot_action == 7:
            bot_y += character_move_speed
            if bot_y + character_size >= disp_y/2 + arena_y/2:
                bot_y = disp_y/2 + arena_y/2 - character_size
            elif agent_y <= bot_y + character_size <= agent_y + character_size:
                bot_y = agent_y - character_size
        elif bot_action == 8:
            bot_x -= character_move_speed
            if bot_x <= disp_x/2 - arena_x/2:
                bot_x = disp_x/2 - arena_x/2
            elif agent_x <= bot_x <= agent_x + character_size:
                bot_x = agent_x + character_size

    if bot_beam_fire == True:
        if beam_hit_detector("bot"):
            print "Agent Got Hit!"
            agent_hp -= beam_damage
            reward += -20
            bot_beam_size_x = bot_beam_size_y = 0
            bot_beam_x = bot_beam_y = beam_ob
            if agent_hp <= 0:
                cont = False
                winner = "Bot"
    if bot_beam_cooldown_active == True:
        bot_beam_cooldown -= 1
        if bot_beam_cooldown == 0:
            bot_beam_cooldown_active = False

    if 1 <= agent_action <= 4:
        if agent_beam_cooldown_active == False:
            agent_beam_fire = True
            agent_beam_cooldown_active = True
            agent_beam_cooldown = beam_cooldown_time
            if agent_action == 1:
                if agent_y > disp_y/2 - arena_y/2:
                    agent_beam_x = agent_x - beam_width/2; agent_beam_y = disp_y/2 - arena_y/2
                    agent_beam_size_x = beam_width; agent_beam_size_y = agent_y - disp_y/2 + arena_y/2
                else:
                    reward += -5
            elif agent_action == 2:
                if agent_x + character_size < disp_x/2 + arena_x/2:
                    agent_beam_x = agent_x + character_size; agent_beam_y = agent_y + character_size/2 - beam_width/2
                    agent_beam_size_x = disp_x/2 + arena_x/2 - agent_x - character_size; agent_beam_size_y = beam_width
                else:
                    reward += -5
            elif agent_action == 3:
                if agent_y + character_size < disp_y/2 + arena_y/2:
                    agent_beam_x = agent_x + character_size/2 - beam_width/2; agent_beam_y = agent_y + character_size
                    agent_beam_size_x = beam_width; agent_beam_size_y = disp_y/2 + arena_y/2 - agent_y - character_size
                else:
                    reward += -5
            elif agent_action == 4:
                if agent_x > disp_x/2 - arena_x/2:
                    agent_beam_x = disp_x/2 - arena_x/2; agent_beam_y = agent_y + character_size/2 - beam_width/2
                    agent_beam_size_x = agent_x - disp_x/2 + arena_x/2; agent_beam_size_y = beam_width
                else:
                    reward += -5
        else:
            reward -= 5

    elif 5 <= agent_action <= 8:
        if agent_action == 5:
            agent_y -= character_move_speed
            if agent_y <= disp_y/2 - arena_y/2:
                agent_y = disp_y/2 - arena_y/2
                reward += -5
            elif bot_y <= agent_y <= bot_y + character_size and bot_x <= agent_x <= bot_x + character_size:
                agent_y = bot_y + character_size
                reward += -1
        elif agent_action == 6:
            agent_x += character_move_speed
            if agent_x + character_size >= disp_x/2 + arena_x/2:
                agent_x = disp_x/2 + arena_x/2 - character_size
                reward += -5
            elif bot_x <= agent_x + character_size <= bot_x + character_size and bot_y <= agent_y <= bot_y + character_size:
                agent_x = bot_x - character_size
                reward += -1
        elif agent_action == 7:
            agent_y += character_move_speed
            if agent_y + character_size >= disp_y/2 + arena_y/2:
                agent_y = disp_y/2 + arena_y/2 - character_size
                reward += -5
            elif bot_y <= agent_y + character_size <= bot_y + character_size and bot_x <= agent_x <= bot_x + character_size:
                agent_y = bot_y - character_size
                reward += -1
        elif agent_action == 8:
            agent_x -= character_move_speed
            if agent_x <= disp_x/2 - arena_x/2:
                agent_x = disp_x/2 - arena_x/2
                reward += -5
            elif bot_x <= agent_x <= bot_x + character_size and bot_y <= agent_y <= bot_y + character_size:
                agent_x = bot_x + character_size
                reward += -1
    if agent_beam_fire == True:
        if beam_hit_detector("agent"):
            print "Bot Got Hit!"
            bot_hp -= beam_damage
            reward += 40
            agent_beam_size_x = agent_beam_size_y = 0
            agent_beam_x = agent_beam_y = beam_ob
            if bot_hp <= 0:
                successful = True
                cont = False
                winner = "Agent"
    if agent_beam_cooldown_active == True:
        agent_beam_cooldown -= 1
        if agent_beam_cooldown == 0:
            agent_beam_cooldown_active = False
    return reward, cont, successful, winner

def bot_beam_dir_detector():
    global bot_current_action
    if bot_current_action == 1:
        bot_beam_dir = 2
    elif bot_current_action == 2:
        bot_beam_dir = 4
    elif bot_current_action == 3:
        bot_beam_dir = 3
    elif bot_current_action == 4:
        bot_beam_dir = 1
    else:
        bot_beam_dir = 0
    return bot_beam_dir

#Parameters
y = 0.75
e = 0.3
num_episodes = 5000
batch_size = 25
complexity = 100

with tf.Session() as sess:
    sess.run(initialize)
    success = 0
    turn_count = 0
    for i in tqdm(range(1, num_episodes)):
        epoch = i
        rAll = 0; d = False; c = True; j = 0
        param_init()
        samples = []
        while c == True:
            j += 1
            turn_count += 1 
            current_state = np.array([[float(agent_x - disp_x/2 + arena_x/2) / float(arena_x),
                                       float(agent_y - disp_y/2 + arena_y/2) / float(arena_y),
                                       float(bot_x - disp_x/2 + arena_x/2) / float(arena_x),
                                       float(bot_y - disp_y/2 + arena_y/2) / float(arena_y),
                                       float(bot_x - agent_x) / float(arena_x),
                                       float(bot_y - agent_y) / float(arena_y),
                                       float(agent_beam_cooldown) / float(beam_cooldown_time),
                                       agent_beam_cooldown_active,
                                       float(bot_beam_cooldown) / float(beam_cooldown_time),
                                       bot_beam_cooldown_active
                                       ]])
            b = bot_take_action()
            if np.random.rand(1) < e:
                a = np.array([random.randint(0, 8)])
            else:
                a, _ = sess.run([predict, Q],feed_dict={input_layer : current_state})
            r, c, d, winner = action(a + 1, b)
            bot_beam_dir = bot_beam_dir_detector()
            next_state = np.array([[float(agent_x - disp_x / 2 + arena_x / 2) / float(arena_x),
                                    float(agent_y - disp_y / 2 +arena_y / 2) / float(arena_y),
                                    float(bot_x - disp_x / 2 + arena_x / 2) / float(arena_x),
                                    float(bot_y - disp_y / 2 + arena_y / 2) / float(arena_y),
                                    float(bot_x - agent_x) / float(arena_x),
                                    float(bot_y - agent_y) / float(arena_y),
                                    float(agent_beam_cooldown) / float(beam_cooldown_time),
                                    agent_beam_cooldown_active,
                                    float(bot_beam_cooldown) / float(beam_cooldown_time),
                                    bot_beam_cooldown_active
                                    ]])
            samples.append([current_state, a, r, next_state, d])
            if len(samples) > batch_size:
                for count in xrange(batch_size):
                    [batch_current_state, action_taken, reward, batch_next_state, terminal] = samples[random.randint(0, len(samples) - 1)]
                    if turn_count % 50 == 0:
                        target_weight_1 = weight_1
                    batch_allQ = sess.run(target_Q, feed_dict={target_input_layer : batch_current_state})
                    batch_Q1 = sess.run(target_Q, feed_dict = {target_input_layer : batch_next_state})
                    batch_maxQ1 = np.max(batch_Q1)
                    batch_targetQ = batch_allQ
                    if np.isnan(batch_targetQ).any():
                        print "NaN Detected. Episode :%d Terminating..." % i
                        sys.exit()
                    if terminal:
                        batch_targetQ[0][action_taken] = reward
                    else:
                        batch_targetQ[0][action_taken] = reward + y * batch_maxQ1
                    sess.run(updateModel, feed_dict={input_layer : batch_current_state, next_Q : batch_targetQ})
            rAll += r
            
            if d == True:
                e *= 0.9
                success += 1
                break
            screen_blit()
            display.update()

        jList.append(j)
        rList.append(rAll)
        if winner == "bot":
            win_lost = -1
        else:
            win_lost = 1
        print "Winner: ", winner, "   Estimated Score: ", calculate_score(rAll, j, win_lost)
        
        if i == 4999:
            plt.plot(jList)
            savefig("Turns.png")
            plt.close()
            plt.plot(rList)
            savefig("Rewards.png")
            plt.close()
            saver = tf.train.Saver()
            saver.save(sess, 'model')
