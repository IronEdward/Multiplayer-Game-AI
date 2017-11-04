from pygame import *
from pygame.locals import *
import sys
from time import sleep
import numpy as np
import random
import tensorflow as tf
from pylab import savefig
import matplotlib.pyplot as plt
from tqdm import tqdm

"""

******Refined Game Program. Now finally working properly!******

------Agent------

Convolutional Neural Nets:               No
Layers:                                  1
Experience Replay:                       Yes
Players:                                 Square
Parameters:                              9
Bullet:                                  Beam
Activation Function:                     Linear
Target Network:                          Yes
"""

arena_size = 500

#Screen Setup
disp_x, disp_y = 1000, 800
arena_x = arena_y = arena_size
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
character_move_speed = 50

#Initialize character stats
character_init_health = 100

#initialize bullet stats
beam_damage = 20
beam_width = 20
beam_ob = -100
beam_cooldown_time = 5


class Agent_Init:
    #The Neural Network
    def __init__(self):
        self.input_layer = tf.placeholder(shape=[1,10],dtype=tf.float32)
        self.weight_1 = tf.Variable(tf.random_uniform([10,9],0,0.01))
        self.Q = tf.matmul(self.input_layer, self.weight_1)
        self.predict = tf.argmax(self.Q, 1)
        self.next_Q = tf.placeholder(shape=[1,9],dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.next_Q - self.Q))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.updateModel = self.trainer.minimize(self.loss)
        self.target_input_layer = tf.placeholder(shape=[1,10],dtype=tf.float32)
        self.target_weight_1 = tf.Variable(tf.random_uniform([10,9],0,0.01))
        self.target_Q = tf.matmul(self.target_input_layer, self.target_weight_1)
        self.target_next_Q = tf.placeholder(shape=[1,9],dtype=tf.float32)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

agent_1 = Agent_Init()
agent_2 = Agent_Init()
initialize = tf.global_variables_initializer()

jList = []
rList = []

init()
font.init()
myfont = font.SysFont('Comic Sans MS', 15)
myfont2 = font.SysFont('Comic Sans MS', 150)
myfont3 = font.SysFont('Gothic', 30)
disp = display.set_mode((disp_x, disp_y), 0, 32)

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
        draw.rect(disp, energy_blue, (bot_beam_x, bot_beam_y, bot_beam_size_x, bot_beam_size_y))
        bot_beam_size_x = bot_beam_size_y = 0
        bot_beam_x = bot_beam_y = beam_ob
        bot_beam_fire = False
    if agent_beam_fire == True:
        draw.rect(disp, green_yellow, (agent_beam_x, agent_beam_y, agent_beam_size_x, agent_beam_size_y))
        agent_beam_size_x = agent_beam_size_y = 0
        agent_beam_x = agent_beam_y = beam_ob
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


def beam_hit_detector(player):
    global agent_x, agent_y, bot_x, bot_y, agent_beam_fire, bot_beam_fire, agent_beam_x, \
    bot_beam_x, agent_beam_y, bot_beam_y, agent_beam_size_x, agent_beam_size_y, \
    bot_beam_size_x, bot_beam_size_y, bot_current_action, agent_current_action, beam_width, character_size

    if player == "bot":
        if bot_current_action == 1:
            return disp_y/2 - arena_y/2 <= agent_y <= bot_y and (agent_x < bot_beam_x + beam_width < agent_x + character_size or agent_x < bot_beam_x < agent_x + character_size)
        elif bot_current_action == 2:
            return bot_x <= agent_x <= disp_x/2 + arena_x/2 and (agent_y < bot_beam_y + beam_width < agent_y + character_size or agent_y < bot_beam_y < agent_y + character_size)
        elif bot_current_action == 3:
            return bot_y <= agent_y <= disp_y/2 + arena_y/2 and (agent_x < bot_beam_x + beam_width < agent_x + character_size or agent_x < bot_beam_x < agent_x + character_size)
        elif bot_current_action == 4:
            return disp_x/2 - arena_x/2 <= agent_x <= bot_x and (agent_y < bot_beam_y + beam_width < agent_y + character_size or agent_y < bot_beam_y < agent_y + character_size)
    else:
        if agent_current_action == 1:
            return disp_y/2 - arena_y/2 <= bot_y <= agent_y and (bot_x < agent_beam_x + beam_width < bot_x + character_size or bot_x < agent_beam_x < bot_x + character_size)
        elif agent_current_action == 2:
            return agent_x <= bot_x <= disp_x/2 + arena_x/2 and (bot_y < agent_beam_y + beam_width < bot_y + character_size or bot_y < agent_beam_y < bot_y + character_size)
        elif agent_current_action == 3:
            return agent_y <= bot_y <= disp_y/2 + arena_y/2 and (bot_x < agent_beam_x + beam_width < bot_x + character_size or bot_x < agent_beam_x < bot_x + character_size) 
        elif agent_current_action == 4:
            return disp_x/2 - arena_x/2 <= bot_x <= agent_x and (bot_y < agent_beam_y + beam_width < bot_y + character_size or bot_y < agent_beam_y < bot_y + character_size)


"""
def mapping(maximum, number):
    return int(number * maximum)
"""
def action(agent_action, bot_action):
    global agent_x, agent_y, bot_x, bot_y, agent_hp, bot_hp, agent_beam_fire, \
    bot_beam_fire, agent_beam_x, bot_beam_x, agent_beam_y, bot_beam_y, agent_beam_size_x, \
    agent_beam_size_y, bot_beam_size_x, bot_beam_size_y, beam_width, agent_current_action, \
    bot_current_action, character_size, agent_beam_cooldown, agent_beam_cooldown_active, \
    bot_beam_cooldown, bot_beam_cooldown_active, beam_cooldown_time

    agent_current_action = agent_action; bot_current_action = bot_action
    reward = reward2 = 0
    cont = True; 
    successful_agent_1 = successful_agent_2 = False
    winner = ""
    if 1 <= bot_action <= 4:
        if bot_beam_cooldown_active == False:
            bot_beam_fire = True
            bot_beam_cooldown_active = True
            bot_beam_cooldown = beam_cooldown_time
            if bot_action == 1:
                if bot_y > disp_y/2 - arena_y/2:
                    bot_beam_x = bot_x + character_size/2 - beam_width/2; bot_beam_y = disp_y/2 - arena_y/2
                    bot_beam_size_x = beam_width; bot_beam_size_y = bot_y - disp_y/2 + arena_y/2
                else:
                    reward2 += -5
            elif bot_action == 2:
                if bot_x + character_size < disp_x/2 + arena_x/2:
                    bot_beam_x = bot_x + character_size; bot_beam_y = bot_y + character_size/2 - beam_width/2
                    bot_beam_size_x = disp_x/2 + arena_x/2 - bot_x - character_size; bot_beam_size_y = beam_width
                else:
                    reward2 += -5
            elif bot_action == 3:
                if bot_y + character_size < disp_y/2 + arena_y/2:
                    bot_beam_x = bot_x + character_size/2 - beam_width/2; bot_beam_y = bot_y + character_size
                    bot_beam_size_x = beam_width; bot_beam_size_y = disp_y/2 + arena_y/2 - bot_y - character_size
                else:
                    reward2 += -5
            elif bot_action == 4:
                if bot_x > disp_x/2 - arena_x/2:
                    bot_beam_x = disp_x/2 - arena_x/2; bot_beam_y = bot_y + character_size/2 - beam_width/2
                    bot_beam_size_x = bot_x - disp_x/2 + arena_x/2; bot_beam_size_y = beam_width
                else:
                    reward2 += -5
        else:
            reward2 -= 5
    if 1 <= agent_action <= 4:
        if agent_beam_cooldown_active == False:
            agent_beam_fire = True
            agent_beam_cooldown_active = True
            agent_beam_cooldown = beam_cooldown_time
            if agent_action == 1:
                if agent_y > disp_y/2 - arena_y/2:
                    agent_beam_x = agent_x + character_size/2 - beam_width/2; agent_beam_y = disp_y/2 - arena_y/2
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
    if bot_beam_fire == True:
        if beam_hit_detector("bot"):
            #print("Agent Got Hit!")
            agent_hp -= beam_damage
            reward += -20
            reward2 += 40
            if agent_hp <= 0:
                cont = False
                successful_agent_2 = True
                winner = "Bot"
    if bot_beam_cooldown_active == True:
        bot_beam_cooldown -= 1
        if bot_beam_cooldown == 0:
            bot_beam_cooldown_active = False

    if agent_beam_fire == True:
        if beam_hit_detector("agent"):
            #print("Bot Got Hit!")
            bot_hp -= beam_damage
            reward += 40
            reward2 -= 20
            if bot_hp <= 0:
                successful_agent_1 = True
                cont = False
                winner = "Agent"
    if agent_beam_cooldown_active == True:
        agent_beam_cooldown -= 1
        if agent_beam_cooldown == 0:
            agent_beam_cooldown_active = False

    if 5 <= bot_action <= 8:
        if bot_action == 5:
            bot_y -= character_move_speed
            if bot_y <= disp_y/2 - arena_y/2:
                bot_y = disp_y/2 - arena_y/2
                reward2 += -5
            elif agent_y <= bot_y <= agent_y + character_size and (agent_x <= bot_x < agent_x + character_size or bot_x <= agent_x < bot_x + character_size):
                bot_y = agent_y + character_size
                reward2 += -1
        elif bot_action == 6:
            bot_x += character_move_speed
            if bot_x >= disp_x/2 + arena_x/2 - character_size:
                bot_x = disp_x/2 + arena_x/2 - character_size
                reward2 += -5
            elif agent_x <= bot_x + character_size <= agent_x + character_size and (agent_y <= bot_y < agent_y + character_size or bot_y <= agent_y < bot_y + character_size):
                bot_x = agent_x - character_size
                reward2 += -1
        elif bot_action == 7:
            bot_y += character_move_speed
            if bot_y + character_size >= disp_y/2 + arena_y/2:
                bot_y = disp_y/2 + arena_y/2 - character_size
                reward2 += -5
            elif agent_y <= bot_y <= agent_y + character_size and (agent_x <= bot_x < agent_x + character_size or bot_x <= agent_x < bot_x + character_size):
                bot_y = agent_y - character_size
                reward2 += -1
        elif bot_action == 8:
            bot_x -= character_move_speed
            if bot_x <= disp_x/2 - arena_x/2:
                bot_x = disp_x/2 - arena_x/2
                reward2 += -5
            elif agent_x <= bot_x <= agent_x + character_size and (agent_y <= bot_y < agent_y + character_size or bot_y <= agent_y < bot_y + character_size):
                bot_x = agent_x + character_size
                reward2 += -1

    if 5 <= agent_action <= 8:
        if agent_action == 5:
            agent_y -= character_move_speed
            if agent_y <= disp_y/2 - arena_y/2:
                agent_y = disp_y/2 - arena_y/2
                reward += -5
            elif bot_y <= agent_y <= bot_y + character_size and (bot_x <= agent_x < bot_x + character_size or agent_x <= bot_x < agent_x + character_size):
                agent_y = bot_y + character_size
                reward += -1
        elif agent_action == 6:
            agent_x += character_move_speed
            if agent_x + character_size >= disp_x/2 + arena_x/2:
                agent_x = disp_x/2 + arena_x/2 - character_size
                reward += -5
            elif bot_x <= agent_x + character_size <= bot_x + character_size and (bot_y <= agent_y <= bot_y + character_size or agent_y <= bot_y < agent_y + character_size):
                agent_x = bot_x - character_size
                reward += -1
        elif agent_action == 7:
            agent_y += character_move_speed
            if agent_y + character_size >= disp_y/2 + arena_y/2:
                agent_y = disp_y/2 + arena_y/2 - character_size
                reward += -5
            elif bot_y <= agent_y + character_size <= bot_y + character_size and (bot_x <= agent_x < bot_x + character_size or agent_x <= bot_x < agent_x + character_size):
                agent_y = bot_y - character_size
                reward += -1
        elif agent_action == 8:
            agent_x -= character_move_speed
            if agent_x <= disp_x/2 - arena_x/2:
                agent_x = disp_x/2 - arena_x/2
                reward += -5
            elif bot_x <= agent_x <= bot_x + character_size and (bot_y <= agent_y <= bot_y + character_size or agent_y <= bot_y < agent_y + character_size):
                agent_x = bot_x + character_size
                reward += -1

    return reward, reward2, cont, successful_agent_1, successful_agent_2, winner


#Parameters
y = 0.75
e1 = e2 = 0.3
num_episodes = 10000
batch_size = 25

agent_1.sess.run(initialize)
agent_2.sess.run(initialize)
turn_count = 0
for i in tqdm(range(1, num_episodes)):
    rAll = 0; d = False; c = True; j = 0
    param_init()
    samples_agent_1 = samples_agent_2 = []
    while c == True:
        j += 1
        turn_count += 1 
        current_state_agent_1 = np.array([[float(agent_x - disp_x/2 + arena_x/2) / float(arena_x),
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
        current_state_agent_2 = np.array([[float(bot_x - disp_x/2 + arena_x/2) / float(arena_x),
                                    float(bot_y - disp_y/2 + arena_y/2) / float(arena_y),
                                    float(agent_x - disp_x/2 + arena_x/2) / float(arena_x),
                                    float(agent_y - disp_y/2 + arena_y/2) / float(arena_y),
                                    float(agent_x - bot_x) / float(arena_x),
                                    float(agent_y - bot_y) / float(arena_y),
                                    float(bot_beam_cooldown) / float(beam_cooldown_time),
                                    bot_beam_cooldown_active,
                                    float(agent_beam_cooldown) / float(beam_cooldown_time),
                                    agent_beam_cooldown_active
                                    ]])
        if np.random.rand(1) < e1:
            a = np.array([random.randint(0, 8)])
        else:
            a, _ = agent_1.sess.run([agent_1.predict, agent_1.Q],feed_dict={agent_1.input_layer : current_state_agent_1})

        if np.random.rand(1) < e2:
            b = np.array([random.randint(0, 8)])
        else:
            b, _ = agent_2.sess.run([agent_2.predict, agent_2.Q],feed_dict={agent_2.input_layer : current_state_agent_2})
        print(a+1)
        r, r2, c, d, d2, winner = action(a + 1, b + 1)
        next_state_agent_1 = np.array([[float(agent_x - disp_x / 2 + arena_x / 2) / float(arena_x),
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
        next_state_agent_2 = np.array([[float(bot_x - disp_x / 2 + arena_x / 2) / float(arena_x),
                                float(bot_y - disp_y / 2 +arena_y / 2) / float(arena_y),
                                float(agent_x - disp_x / 2 + arena_x / 2) / float(arena_x),
                                float(agent_y - disp_y / 2 + arena_y / 2) / float(arena_y),
                                float(agent_x - bot_x) / float(arena_x),
                                float(agent_y - bot_y) / float(arena_y),
                                float(bot_beam_cooldown) / float(beam_cooldown_time),
                                bot_beam_cooldown_active,
                                float(agent_beam_cooldown) / float(beam_cooldown_time),
                                agent_beam_cooldown_active
                                ]])
        samples_agent_1.append([current_state_agent_1, a, r, next_state_agent_1, d])
        samples_agent_2.append([current_state_agent_2, b, r2, next_state_agent_2, d2])
        if len(samples_agent_1) > batch_size:
            for count in range(batch_size):
                [batch_current_state, action_taken, reward, batch_next_state, terminal] = samples_agent_1[random.randint(0, len(samples_agent_1) - 1)]
                if turn_count % 50 == 0:
                    agent_1.target_weight_1 = agent_1.weight_1
                batch_allQ = agent_1.sess.run(agent_1.target_Q, feed_dict={agent_1.target_input_layer : batch_current_state})
                batch_Q1 = agent_1.sess.run(agent_1.target_Q, feed_dict = {agent_1.target_input_layer : batch_next_state})
                batch_maxQ1 = np.max(batch_Q1)
                batch_targetQ = batch_allQ
                if np.isnan(batch_targetQ).any():
                    print("NaN Detected. Episode :%d Terminating..." % i)
                    sys.exit()
                if terminal:
                    batch_targetQ[0][action_taken] = reward
                else:
                    batch_targetQ[0][action_taken] = reward + y * batch_maxQ1
                agent_1.sess.run(agent_1.updateModel, feed_dict={agent_1.input_layer : batch_current_state, agent_1.next_Q : batch_targetQ})
                #print "Loss: ", sess.run(agent_1.loss, feed_dict={agent_1.input_layer : batch_current_state, agent_1.next_Q : batch_targetQ})
        if len(samples_agent_2) > batch_size:
            for count in range(batch_size):
                [batch_current_state, action_taken, reward, batch_next_state, terminal] = samples_agent_2[random.randint(0, len(samples_agent_2) - 1)]
                if turn_count % 50 == 0:
                    agent_2.target_weight_1 = agent_2.weight_1
                batch_allQ = agent_2.sess.run(agent_2.target_Q, feed_dict={agent_2.target_input_layer : batch_current_state})
                batch_Q1 = agent_2.sess.run(agent_2.target_Q, feed_dict = {agent_2.target_input_layer : batch_next_state})
                batch_maxQ1 = np.max(batch_Q1)
                batch_targetQ = batch_allQ
                if np.isnan(batch_targetQ).any():
                    print("NaN Detected. Episode :%d Terminating..." % i)
                    sys.exit()
                if terminal:
                    batch_targetQ[0][action_taken] = reward
                else:
                    batch_targetQ[0][action_taken] = reward + y * batch_maxQ1
                agent_2.sess.run(agent_2.updateModel, feed_dict={agent_2.input_layer : batch_current_state, agent_2.next_Q : batch_targetQ})
                #print "Loss: ", sess.run(agent_1.loss, feed_dict={agent_1.input_layer : batch_current_state, agent_1.next_Q : batch_targetQ})
        for eve in event.get():
            if eve.type==QUIT:
                quit()
                sys.exit()
        keys = key.get_pressed()
        if keys[K_s]:
            screen_blit()
            display.update()
        else:
            disp.fill(black)
            display.update()

    jList.append(j)
    if d == True:
        e1 *= 0.9
    else:
        e2 *= 0.9
    print("Winner: ", winner)
    if i%1000 == 0:
        agent_1.saver.save(agent_1.sess, 'agent_1')
        agent_2.saver.save(agent_2.sess, 'agent_2')
