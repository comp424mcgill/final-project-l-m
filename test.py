from itertools import combinations_with_replacement

max_step = 2
old_pos = (2,2)

out = set()

# generate valid moves [,]
for k in range(1,max_step+1):
    # get random combination
    poss_paths = combinations_with_replacement(["U","D","L","R"], k)
    #poss_paths = random.shuffle(poss_paths)

    for path in poss_paths:
        end_pos = old_pos # var declaration

        for dir in path:
            if dir == "U": #x-1
                new_pos = (end_pos[0]-1,end_pos[1])
                end_pos = new_pos

            elif dir == "D": #x+1
                new_pos = (end_pos[0]+1,end_pos[1])
                end_pos = new_pos  
            
            elif dir == "L": #y-1
                new_pos = (end_pos[0],end_pos[1]-1)
                end_pos = new_pos
            
            elif dir == "R": #y+1
                new_pos = (end_pos[0],end_pos[1]+1)
                end_pos = new_pos

        # if end_pos not in out:
        #     out.append(end_pos)

        
        for dir in range(4):
            if (end_pos,dir) not in out:
                out.add((end_pos,dir))
print(len(out))
print(out)

#good:
    # [(1, 2), (3, 2), (2, 1), (2, 3), (0, 2), (2, 2), (1, 1), (1, 3), (4, 2), (3, 1), (3, 3), (2, 0), (2, 4)]

