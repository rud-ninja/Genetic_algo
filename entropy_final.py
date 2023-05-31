# Genetic evolution simulator


import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import textwrap
import warnings
warnings.filterwarnings('ignore')

random.seed(9)

def sigmoid(array):
    sigmoid_array = 1 / (1 + np.exp(-array))
    return sigmoid_array

class neural_network():
  def __init__(self, coordinates):
    self.position = random.choice(coordinates)
    coordinates.remove(self.position)
    self.starting_pos = self.position
    
    self.input_vector = []
    self.gene_hex = []
    self.gene_bin = []
    self.last_x_mov = 0
    self.last_y_mov = 0
    self.displacement = 0
    self.xsteps = 0
    self.ysteps = 0

  def make_gene_sequence(self, gene_hex, gene_length):
    self.gene_hex = []
    for _ in range(gene_length):
      random_genome = random.choices([str(x) for x in list(range(10))]+['a', 'b', 'c', 'd', 'e', 'f'], k=8)
      self.gene_hex.append(''.join(random_genome))

  def convert_gene_binary(self, gene_hex, gene_bin):
    self.gene_bin = []
    for gene in self.gene_hex:
      bin_genome = []
      for g in gene:
        seg = bin(int(g, 16))[2:].zfill(4)
        bin_genome+=seg
      self.gene_bin.append(''.join(bin_genome))

  def make_connections(self, gene_bin):
    n_input = 25
    n_hidden = 2
    n_output = 15
    div = 9000

    self.input_hidden_connection = np.zeros((n_input, n_hidden))
    self.hidden_output_connection = np.zeros((n_hidden, n_output))
    self.hidden_hidden_connection = np.zeros((n_hidden, n_hidden))
    self.input_output_connection = np.zeros((n_input, n_output))

    self.input_hidden_weight = np.zeros((n_input, n_hidden))
    self.hidden_output_weight = np.zeros((n_hidden, n_output))
    self.hidden_hidden_weight = np.zeros((n_hidden, n_hidden))
    self.input_output_weight = np.zeros((n_input, n_output))

    self.hidden_state = np.zeros((1, n_hidden))

    for connection in self.gene_bin:
      source = int(connection[0])
      source_id = ''.join(connection[1:8])
      sink = int(connection[8])
      sink_id = ''.join(connection[9:16])
      connection_weight = ''.join(connection[16:])

      if source==0 and sink==1:
        source_id = int(source_id, 2)%n_input
        sink_id = int(sink_id, 2)%n_hidden
        self.input_hidden_connection[source_id][sink_id] = 1
        self.input_hidden_weight[source_id][sink_id] = (int(connection_weight, 2) - 2**16 if int(connection_weight[0]) else int(connection_weight, 2))/div

      if source==1 and sink==0:
        source_id = int(source_id, 2)%n_hidden
        sink_id = int(sink_id, 2)%n_output
        self.hidden_output_connection[source_id][sink_id] = 1
        self.hidden_output_weight[source_id][sink_id] = (int(connection_weight, 2) - 2**16 if int(connection_weight[0]) else int(connection_weight, 2))/div

      if source==0 and sink==0:
        source_id = int(source_id, 2)%n_input
        sink_id = int(sink_id, 2)%n_output
        self.input_output_connection[source_id][sink_id] = 1
        self.input_output_weight[source_id][sink_id] = (int(connection_weight, 2) - 2**16 if int(connection_weight[0]) else int(connection_weight, 2))/div

      if source==1 and sink==1:
        source_id = int(source_id, 2)%n_hidden
        sink_id = int(sink_id, 2)%n_hidden
        self.hidden_hidden_connection[source_id][sink_id] = 1
        self.hidden_hidden_weight[source_id][sink_id] = (int(connection_weight, 2) - 2**16 if int(connection_weight[0]) else int(connection_weight, 2))/div
  
  def forward_pass(self):  
    self.hidden_layer_output = np.tanh(np.dot(np.transpose(self.input_vector), self.input_hidden_weight) + self.hidden_state)
    # self.hidden_layer_output = sigmoid(np.dot(np.transpose(self.input_vector), self.input_hidden_weight) + self.hidden_state)
    self.hidden_state = np.dot(self.hidden_layer_output, self.hidden_hidden_weight) * 0.01
    self.output_layer_output = np.tanh(np.dot(np.transpose(self.input_vector), self.input_output_weight) + np.dot(self.hidden_layer_output, self.hidden_output_weight))
    # self.output_layer_output = sigmoid(np.dot(np.transpose(self.input_vector), self.input_output_weight) + np.dot(self.hidden_layer_output, self.hidden_output_weight))


    # # for sigmoid activation function
    # output_sum = sum(self.output_layer_output[0])
    # normalised_output = [x/output_sum for x in self.output_layer_output[0]]
    # self.reaction = np.argmax(normalised_output)
   


    # for hyperbolic tangent activation function
    softmax_outputs = np.exp(self.output_layer_output) / np.sum(np.exp(self.output_layer_output))
    self.reaction = np.argmax(softmax_outputs[0])





# input sensory functions

def dist_e(boundary, position):
  return abs(boundary-position[0])

def dist_w(boundary, position):
  return abs(-boundary-position[0])

def dist_n(boundary, position):
  return abs(boundary-position[1])

def dist_s(boundary, position):
  return abs(-boundary-position[1])

def dist_ne(boundary, position):
  dist_x = boundary-position[0]
  dist_y = boundary-position[1]
  dist_sq = dist_x**2 + dist_y**2
  return dist_sq**0.5

def dist_nw(boundary, position):
  dist_x = -boundary-position[0]
  dist_y = boundary-position[1]
  dist_sq = dist_x**2 + dist_y**2
  return dist_sq**0.5

def dist_se(boundary, position):
  dist_x = boundary-position[0]
  dist_y = -boundary-position[1]
  dist_sq = dist_x**2 + dist_y**2
  return dist_sq**0.5

def dist_sw(boundary, position):
  dist_x = -boundary-position[0]
  dist_y = -boundary-position[1]
  dist_sq = dist_x**2 + dist_y**2
  return dist_sq**0.5

def dist_ctr(position):
  dist_sq = position[0]**2+position[1]**2
  return dist_sq**0.5

def e_occupied(position, org_list, restricted):
  temp = 0
  east_pos = [position[0]+1, position[1]]
  if (east_pos in [k.position for k in org_list]) or (east_pos in restricted):
    temp = 1

  return temp

def w_occupied(position, org_list, restricted):
  temp = 0
  west_pos = [position[0]-1, position[1]]
  if (west_pos in [k.position for k in org_list]) or (west_pos in restricted):
    temp = 1
    
  return temp

def n_occupied(position, org_list, restricted):
  temp = 0
  north_pos = [position[0], position[1]+1]
  if (north_pos in [k.position for k in org_list]) or (north_pos in restricted):
    temp = 1
    
  return temp

def s_occupied(position, org_list, restricted):
  temp = 0
  south_pos = [position[0], position[1]-1]
  if (south_pos in [k.position for k in org_list]) or (south_pos in restricted):
    temp = 1
    
  return temp

def ne_occupied(position, org_list, restricted):
  temp = 0
  ne_pos = [position[0]+1, position[1]+1]
  if (ne_pos in [k.position for k in org_list]) or (ne_pos in restricted):
    temp = 1
    
  return temp

def nw_occupied(position, org_list, restricted):
  temp = 0
  nw_pos = [position[0]-1, position[1]+1]
  if (nw_pos in [k.position for k in org_list]) or (nw_pos in restricted):
    temp = 1
    
  return temp

def se_occupied(position, org_list, restricted):
  temp = 0
  se_pos = [position[0]+1, position[1]-1]
  if (se_pos in [k.position for k in org_list]) or (se_pos in restricted):
    temp = 1
    
  return temp

def sw_occupied(position, org_list, restricted):
  temp = 0
  sw_pos = [position[0]-1, position[1]-1]
  if (sw_pos in [k.position for k in org_list]) or (sw_pos in restricted):
    temp = 1
    
  return temp

def pop_dense_ne(org_list, position, boundary):
  ctr = 0
  x_dist = boundary-position[0]
  y_dist = boundary-position[1]
  area = x_dist*y_dist
  for org in org_list:
    if org.position[0]>position[0] and org.position[1]>position[1]:
      ctr+=1
  
  if area!=0:
    density = ctr/area
  elif area==0:
    density = 0
  return density

def pop_dense_nw(org_list, position, boundary):
  ctr = 0
  x_dist = abs(-boundary-position[0])
  y_dist = boundary-position[1]
  area = x_dist*y_dist
  for org in org_list:
    if org.position[0]<position[0] and org.position[1]>position[1]:
      ctr+=1

  if area!=0:
    density = ctr/area
  elif area==0:
    density = 0
  return density

def pop_dense_se(org_list, position, boundary):
  ctr = 0
  x_dist = boundary-position[0]
  y_dist = abs(-boundary-position[1])
  area = x_dist*y_dist
  for org in org_list:
    if org.position[0]>position[0] and org.position[1]<position[1]:
      ctr+=1
  
  if area!=0:
    density = ctr/area
  elif area==0:
    density = 0
  return density

def pop_dense_sw(org_list, position, boundary):
  ctr = 0
  x_dist = abs(-boundary-position[0])
  y_dist = abs(-boundary-position[1])
  area = x_dist*y_dist
  for org in org_list:
    if org.position[0]<position[0] and org.position[1]<position[1]:
      ctr+=1
  
  if area!=0:
    density = ctr/area
  elif area==0:
    density = 0
  return density

def pop_dense_forward(org_list, agent, boundary):
  forward_path = []
  x = agent.position[0]
  y = agent.position[1]
  for i in range(7):
    x+=agent.last_x_mov
    y+=agent.last_y_mov
    forward_path.append([x, y])

  forward_path = [k for k in forward_path if ((k[0]>-boundary and k[0]<boundary) and (k[1]>-boundary and k[1]<boundary))]

  ctr = 0
  for org in org_list:
    if org.position in forward_path:
      ctr+=1

  if len(forward_path)!=0:
    density = ctr/len(forward_path)
  else:
    density = 0

  return density

def nearest_boundary(boundary, position):
  distance_list = [dist_e(boundary, position), dist_w(boundary, position), dist_n(boundary, position), dist_s(boundary, position)]
  distance_list = np.array(distance_list)

  return np.argmax(distance_list)+1


# output motor functions

def move_ctr(boundary, position):
  if position[0]==0 and position[1]==0:
    return position
  
  else:
    new_position = []
    if abs(position[0])>abs(position[1]):
      if position[0]>0:
        new_position.append(position[0]-1)
        new_position.append(position[1])
      if position[0]<0:
        new_position.append(position[0]+1)
        new_position.append(position[1])

    if abs(position[0])<abs(position[1]):
      if position[1]>0:
        new_position.append(position[0])
        new_position.append(position[1]-1)
      if position[1]<0:
        new_position.append(position[0])
        new_position.append(position[1]+1)

    if abs(position[0])==abs(position[1]):
      if position[0]>0 and position[1]>0:
        new_position.append(position[0]-1)
        new_position.append(position[1]-1)
      if position[0]>0 and position[1]<0:
        new_position.append(position[0]-1)
        new_position.append(position[1]+1)
      if position[0]<0 and position[1]>0:
        new_position.append(position[0]+1)
        new_position.append(position[1]-1)
      if position[0]<0 and position[1]<0:
        new_position.append(position[0]+1)
        new_position.append(position[1]+1)

    return new_position

def right_left_turn(center, forward):
  points = []

  if (forward[0]-center[0])!=0 and ((forward[1]-center[1])/(forward[0]-center[0]))!=0:
    slope1 = (forward[1]-center[1])/(forward[0]-center[0])
    slope2 = int(-1/slope1)
    dist = round(((forward[0]-center[0])**2 + (forward[1]-center[1])**2)**0.5)

    for i in range(-10,11):
      for j in range(-10,11):
        ndist = round(((i-center[0])**2 + (j-center[1])**2)**0.5)
        eqval = j-center[1] - (slope2*(i-center[0]))

        if ndist==dist and eqval==0:
          points.append([i,j])

  elif (forward[0]-center[0])==0:
    dist = abs(round(forward[1]-center[1]))
    
    for i in range(-10,11):
      ndist = abs(round(i-center[0]))
      if ndist==dist:
        points.append([i, center[1]])

  elif ((forward[1]-center[1])/(forward[0]-center[0]))==0:
    dist = abs(round(forward[0]-center[0]))
    
    for i in range(-10,11):
      ndist = abs(round(i-center[1]))
      if ndist==dist:
        points.append([center[0], i])

  turns = {}

  for point in points:
    angle1 = math.atan2(forward[1] - center[1], forward[0] - center[0])
    angle2 = math.atan2(point[1] - center[1], point[0] - center[0])

    theta = angle2 - angle1

    if theta < 0:
      theta += 2 * math.pi

    angle = math.degrees(theta)
    if angle==270:
      turns['right'] = point
    if angle==90:
      turns['left'] = point

  return turns

def move_e(boundary, position):
  new_position = []
  if position[0]<boundary-1:
    new_position.append(position[0]+1)
    new_position.append(position[1])
    return new_position
  else:
    return position

def move_w(boundary, position):
  new_position = []
  if position[0]>-boundary+1:
    new_position.append(position[0]-1)
    new_position.append(position[1])
    return new_position
  else:
    return position

def move_n(boundary, position):
  new_position = []
  if position[1]<boundary-1:
    new_position.append(position[0])
    new_position.append(position[1]+1)
    return new_position
  else:
    return position

def move_s(boundary, position):
  new_position = []
  if position[1]>-boundary+1:
    new_position.append(position[0])
    new_position.append(position[1]-1)
    return new_position
  else:
    return position

def move_ne(boundary, position):
  new_position = []
  if position[0]<boundary-1 and position[1]<boundary-1:
    new_position.append(position[0]+1)
    new_position.append(position[1]+1)
    return new_position
  else:
    return position

def move_nw(boundary, position):
  new_position = []
  if position[0]>-boundary+1 and position[1]<boundary-1:
    new_position.append(position[0]-1)
    new_position.append(position[1]+1)
    return new_position
  else:
    return position

def move_se(boundary, position):
  new_position = []
  if position[0]<boundary-1 and position[1]>-boundary+1:
    new_position.append(position[0]+1)
    new_position.append(position[1]-1)
    return new_position
  else:
    return position

def move_sw(boundary, position):
  new_position = []
  if position[0]>-boundary+1 and position[1]>-boundary+1:
    new_position.append(position[0]-1)
    new_position.append(position[1]-1)
    return new_position
  else:
    return position

def move_none(boundary, position):
  return position

def move_forward(boundary, agent):
  new_position = []
  new_position.append(agent.position[0]+agent.last_x_mov)
  new_position.append(agent.position[1]+agent.last_y_mov)

  if (new_position[0]>-boundary and new_position[0]<boundary) and (new_position[1]>-boundary and new_position[1]<boundary):
    return new_position
  else:
    return agent.position

def move_backward(boundary, agent):
  new_position = []
  new_position.append(agent.position[0]-agent.last_x_mov)
  new_position.append(agent.position[1]-agent.last_y_mov)

  if (new_position[0]>-boundary and new_position[0]<boundary) and (new_position[1]>-boundary and new_position[1]<boundary):
    return new_position
  else:
    return agent.position

def turn_right(boundary, agent):
  new_position = []
  center = agent.position
  forward = move_forward(boundary, agent)
  if forward!=center:
    turning_points = right_left_turn(center, forward)
    try:
      new_position = turning_points['right']
    except:
      pass

  if len(new_position) and (new_position[0]>-boundary and new_position[0]<boundary) and (new_position[1]>-boundary and new_position[1]<boundary):
    return new_position
  else:
    return agent.position
  
def turn_left(boundary, agent):
  new_position = []
  center = agent.position
  forward = move_forward(boundary, agent)
  if forward!=center:
    turning_points = right_left_turn(center, forward)
    try:
      new_position = turning_points['left']
    except:
      pass

  if len(new_position) and (new_position[0]>-boundary and new_position[0]<boundary) and (new_position[1]>-boundary and new_position[1]<boundary):
    return new_position
  else:
    return agent.position

def move_rnd(boundary, agent):
  new_position = []
  way = random.choice([1,0])
  if way==1:
    new_position = random.choice([move_e, move_w, move_n, move_s, move_ne, move_nw, move_se, move_sw])(boundary, agent.position)
  if way==0:
    new_position = random.choice([move_forward, move_backward, turn_right, turn_left])(boundary, agent)

  return new_position


movement_dict1 = {0: move_e, 1: move_w, 2: move_n, 3: move_s, 4: move_ne, 5: move_nw, 6: move_se, 7: move_sw, 8: move_none, 9: move_ctr}
movement_dict2 = {10: move_forward, 11: move_backward, 12: turn_right, 13: turn_left, 14: move_rnd}





def selection_function(organisms, limit):
  pool = []
  for org in organisms:
    if abs(org.position[0])>limit:
      pool.append(org)
  return pool

def jaccard_similarity(p1, p2):
  seq1 = ''.join(p1)
  seq2 = ''.join(p2)
  
  kmer1 = []
  kmer2 = []

  for i in range(len(seq1)-2):
    kmer1.append(seq1[i]+seq1[i+1]+seq1[i+2])
    kmer2.append(seq2[i]+seq2[i+1]+seq2[i+2])

  intersection = set(kmer1) & set(kmer2)
  union = set(kmer1) | set(kmer2)
  similarity = len(intersection)/len(union)

  return similarity

def running_stats(samples):
  m = 0
  S = 0
  for i in range(len(samples)):
    x = samples[i]
    old_m = m
    m = m + (x-m)/(i+1)
    S = S + (x-m)*(x-old_m)
  st_dev = round((S/(len(samples)-1))**0.5, 2)
  print(f'Standard deviation: {st_dev}')
  return st_dev

def make_next_population(gene_contributors, population, gene_length, coordinates):
  new_population = []
  mutation_checker = 0
  weights = []
  mutation_rate = 0.001

  for item in gene_contributors:
    net_steps = (item.xsteps**2 + item.ysteps**2)**0.5
    if net_steps>0:
      weights.append(item.displacement/net_steps)
    else:
      weights.append(0.001)


  weights = np.array(weights)/np.sum(np.array(weights))
  weights = weights*population
  contributor = 0
  while len(new_population)<population:
    if contributor==len(gene_contributors):
      contributor=0

    while True:
      parent1 = random.choices(gene_contributors, weights=weights, k=1)[0]
      parent2 = gene_contributors[contributor]
      if parent1==parent2:
        continue
      else:
        break


    genetic_material = [str(x) for x in list(range(10))]+['a', 'b', 'c', 'd', 'e', 'f']
    which_child = random.choice(['either', 'or'])

    if which_child=='either':
      child1 = neural_network(coordinates)

      p1 = list(''.join(parent1.gene_hex))
      p2 = list(''.join(parent2.gene_hex))

      crossover_points = random.choices(range(len(p1)), k=2)
      crossover_points.sort()

      p1[crossover_points[0]:crossover_points[1]] = p2[crossover_points[0]:crossover_points[1]]

      p1 = textwrap.wrap(''.join(p1), 8)

      child1.gene_hex = p1

      mutation_ch1 = random.choices([0, 1], weights=[(1.0-mutation_rate)*population, mutation_rate*population], k=1)
      mutation_checker+=mutation_ch1[0]
      if mutation_ch1[0]==1:
        n_mutations = random.choice(list(range(1,int(gene_length/100)+2)))
        for _ in range(n_mutations):
          random_genome_id = random.choice(range(len(child1.gene_hex)))
          random_mutation_element_id = random.choice(range(len(child1.gene_hex[random_genome_id])))

          list1 = list(child1.gene_hex[random_genome_id])
          list1[random_mutation_element_id] = random.choice([x for x in genetic_material if x!=list1[random_mutation_element_id]])

          list1_comb = ''.join(list1)
          child1.gene_hex[random_genome_id] = list1_comb

      child1.convert_gene_binary(child1.gene_hex, child1.gene_bin)

      child1.make_connections(child1.gene_bin)

      if np.all(child1.input_output_connection==0) and np.all(child1.hidden_output_connection==0):
        pass
      else:
        new_population.append(child1)
  

    if which_child=='or':
      child2 = neural_network(coordinates)

      p1 = list(''.join(parent1.gene_hex))
      p2 = list(''.join(parent2.gene_hex))

      crossover_points = random.choices(range(len(p1)), k=2)
      crossover_points.sort()

      p2[crossover_points[0]:crossover_points[1]] = p1[crossover_points[0]:crossover_points[1]]

      p2 = textwrap.wrap(''.join(p2), 8)

      child2.gene_hex = p2

      mutation_ch2 = random.choices([0, 1], weights=[(1.0-mutation_rate)*population, mutation_rate*population], k=1)
      mutation_checker+=mutation_ch2[0]
      if mutation_ch2[0]==1:
        n_mutations = random.choice(list(range(1,int(gene_length/100)+2)))
        for _ in range(n_mutations):
          random_genome_id = random.choice(range(len(child2.gene_hex)))
          random_mutation_element_id = random.choice(range(len(child2.gene_hex[random_genome_id])))

          list1 = list(child2.gene_hex[random_genome_id])
          list1[random_mutation_element_id] = random.choice([x for x in genetic_material if x!=list1[random_mutation_element_id]])

          list1_comb = ''.join(list1)
          child2.gene_hex[random_genome_id] = list1_comb

      child2.convert_gene_binary(child2.gene_hex, child2.gene_bin)

      child2.make_connections(child2.gene_bin)
      
      if np.all(child2.input_output_connection==0) and np.all(child2.hidden_output_connection==0):
        pass
      else:
        new_population.append(child2)

    contributor+=1

  if int(bool(mutation_checker))==1:
    print(f'mutations occurred: {mutation_checker}')

  return new_population, int(bool(mutation_checker))




borderline = 45
starting_population = 1000 
gene_length = 10
lifetime = 200
barrier = 1
barrier_width = 2
barrier_height = int(borderline/2)
checkpoint = 25
offset = 1

tolerance = 0.2
stopping_limit = 10

while True:
  if int(borderline*2*offset)>=int(starting_population/2):
    break
  else:
    offset+=1


margin = int(borderline-(offset+1))

organisms = []

coordinates = []
restricted_coordinates = []
similarity = []

for i in range(-borderline+1, borderline):
  for j in range(-borderline+1, borderline):
    coordinates.append([i, j])

if barrier==1:
  restr_xcoor = list(range(int(-borderline/2)-barrier_width, int(-borderline/2)+1))+list(range(int(borderline/2), int(borderline/2)+barrier_width+1))
  restr_ycoor = list(range(-barrier_height-int(borderline/5), -int(borderline/5)+1))+list(range(int(borderline/5), int(borderline/5)+barrier_height+1))

  for x in restr_xcoor:
    for y in restr_ycoor:
      restricted_coordinates.append([x, y])

  coordinates = [x for x in coordinates if x not in restricted_coordinates]

random.shuffle(coordinates)

for _ in range(starting_population):
  org = neural_network(coordinates)
  organisms.append(org)



for org in organisms:
  while True:
    org.make_gene_sequence(org.gene_hex, gene_length)
    org.convert_gene_binary(org.gene_hex, org.gene_bin)
    org.make_connections(org.gene_bin)

    if np.all(org.input_output_connection==0) and np.all(org.hidden_output_connection==0):
      continue
    else:
      break




generation = 1
mid = 0
sc = 0
graph = 0
stopping_criteria = False
old_spread = 0
genr = []
surv_rate = []
mut_gen = []
stdev = []
base_path = r"C:\Users\hp\OneDrive\Desktop\Indranuj_Banerjee"

while stopping_criteria==False:


  if generation==1 or mid==3 or graph==checkpoint:
    foldername = f"generation_{generation}"
    os.mkdir(os.path.join(base_path, foldername))

  
  for i in range(lifetime):

    if barrier==1 and (generation==1 or mid==3 or graph==checkpoint):
      fig, ax = plt.subplots()

    for organism in organisms:
      
      if barrier==0:

        if generation==1 or mid==3 or graph==checkpoint:
          plt.scatter(organism.position[0], organism.position[1], s=3)
          plt.axvline(x=margin, ymin=-borderline, ymax=borderline, linewidth=1)
          plt.axvline(x=-margin, ymin=-borderline, ymax=borderline, linewidth=1)
          plt.title('Generation '+str(generation))
          plt.xlim(-borderline,borderline)
          plt.ylim(-borderline,borderline)
          plt.xticks([])
          plt.yticks([])


      if barrier==1:
        if generation==1 or mid==3 or graph==checkpoint:
          ax.scatter(organism.position[0], organism.position[1], s=3)
          ax.axvline(x=margin, ymin=-borderline, ymax=borderline, linewidth=1)
          ax.axvline(x=-margin, ymin=-borderline, ymax=borderline, linewidth=1)
          ax.set_xlim(-borderline, borderline)
          ax.set_ylim(-borderline, borderline)
          rect1 = Rectangle((-borderline/2-barrier_width, -barrier_height-borderline/5), barrier_width, barrier_height)
          rect2 = Rectangle((-borderline/2-barrier_width, borderline/5), barrier_width, barrier_height)
          rect3 = Rectangle((borderline/2, -barrier_height-borderline/5), barrier_width, barrier_height)
          rect4 = Rectangle((borderline/2, borderline/5), barrier_width, barrier_height)
          ax.add_patch(rect1)
          ax.add_patch(rect2)
          ax.add_patch(rect3)
          ax.add_patch(rect4)
          ax.set_title('Generation '+str(generation))
          plt.xticks([])
          plt.yticks([])

      organism.input_vector = [dist_e(borderline, organism.position), dist_w(borderline, organism.position), dist_n(borderline, organism.position), dist_s(borderline, organism.position),
                      dist_ne(borderline, organism.position), dist_nw(borderline, organism.position), dist_se(borderline, organism.position), dist_sw(borderline, organism.position),
                      dist_ctr(organism.position), organism.last_x_mov, organism.last_y_mov,
                      pop_dense_ne(organisms, organism.position, borderline), pop_dense_nw(organisms, organism.position, borderline),
                      pop_dense_se(organisms, organism.position, borderline), pop_dense_sw(organisms, organism.position, borderline),
                      e_occupied(organism.position, organisms, restricted_coordinates), w_occupied(organism.position, organisms, restricted_coordinates), n_occupied(organism.position, organisms, restricted_coordinates), s_occupied(organism.position, organisms, restricted_coordinates),
                      ne_occupied(organism.position, organisms, restricted_coordinates), 
                      nw_occupied(organism.position, organisms, restricted_coordinates), se_occupied(organism.position, organisms, restricted_coordinates), 
                      sw_occupied(organism.position, organisms, restricted_coordinates), pop_dense_forward(organisms, organism, borderline), nearest_boundary(borderline, organism.position)]

      organism.input_vector = np.array(organism.input_vector).reshape(-1,1)

      organism.forward_pass()
      
      old_pos = organism.position
      
      if organism.reaction<=9:
        new_pos = movement_dict1[organism.reaction](borderline, organism.position)
      else:
        new_pos = movement_dict2[organism.reaction](borderline, organism)
      
      if new_pos!=old_pos:
        if (new_pos in [x.position for x in organisms]) or (new_pos in restricted_coordinates):
          new_pos = old_pos
        else:
          organism.position = new_pos

      

      organism.last_x_mov = new_pos[0] - old_pos[0]
      organism.xsteps+=organism.last_x_mov
      organism.last_y_mov = new_pos[1] - old_pos[1]
      organism.ysteps+=organism.last_y_mov

      if i==lifetime-1:
        disp_sq = (organism.position[0] - organism.starting_pos[0])**2 + (organism.position[1] - organism.starting_pos[1])**2
        organism.displacement = disp_sq**0.5


    if generation==1 or mid==3 or graph==checkpoint:
      plt.savefig(os.path.join(base_path, foldername, f"{(i+1):004}"), dpi=100, facecolor='white')
      plt.close()


  next_population_gene_contributors = selection_function(organisms, margin)
  current_survival_rate = (len(next_population_gene_contributors)/starting_population)*100

  if mid<=3 and current_survival_rate>70:
    mid+=1
  else:
    if mid==4:
      pass
    else:
      mid=0
  

  print('Survival percentage of generation',generation,'=',round(current_survival_rate,2))
  genr.append(generation)
  surv_rate.append(current_survival_rate)

  if generation%checkpoint==0:
    spread = running_stats(surv_rate)
    stdev.append(spread)
    if (old_spread - spread)>0 and (old_spread - spread)<=tolerance:
      sc+=1
    else:
      sc = 0
    old_spread = spread
  
  if sc==(stopping_limit-1):
    graph+=1

  if sc==stopping_limit:
    stopping_criteria = True

  if stopping_criteria==False:
    
    coordinates = []

    for i in range(-borderline+1, borderline):
      for j in range(-borderline+1, borderline):
        coordinates.append([i, j])

    if barrier==1:
      coordinates = [x for x in coordinates if x not in restricted_coordinates]

    random.shuffle(coordinates)
    organisms, mutation = make_next_population(next_population_gene_contributors, starting_population, gene_length, coordinates)
    
  mut_gen.append(mutation)
  generation+=1

final_mutation_list = []
for j in range(len(genr)):
  final_mutation_list.append([genr[j], surv_rate[j]*mut_gen[j]])

final_mutation_list = [x for x in final_mutation_list if x[1]>0]

plt.figure(figsize=(15,6))
plt.plot(genr, surv_rate, label='Survival rate', color='darkorchid')
plt.scatter([x[0] for x in final_mutation_list], [x[1] for x in final_mutation_list], label='Mutation', s=15, color='firebrick', alpha=0.7)
plt.title(f"Generation vs Survival rate, Population size: {starting_population}, Gene length: {gene_length}, Lifetime: {lifetime} iterations")
plt.xlabel('Generation')
plt.ylabel('Survival %')
plt.legend()
plt.grid()
plt.savefig(os.path.join(base_path, "final_graph"), dpi=100, facecolor='white')
plt.close()

plt.plot(stdev)
plt.grid()
plt.show()