import time,random,math, ast
import matplotlib.pyplot as plt

amountofdatapoints = 1
gen = 0
class ai:

    def spacer(amount):
        for i in range(amount):
            print("")

    def basedbotcreator(bb,acc,amount):
        
        bots = []
        
        change = int((abs(acc-1)/2) * (100000))
        
        for i in range(amount):
            bot = [0,bb[1]]

            for ii in range(len(bb)-2):
                localbot = []
                for e in range(len(bb[ii+2])):
                    
                    num = bb[ii+2][e] + random.randint(-change,change)/100000
                    if num > 1:
                        num = 1
                    if num < -1:
                        num = -1
                    localbot.append(num)
                bot.append(localbot)
            bots.append(bot)
            
        
        
        return bots

    def botcreator(inn,hid,out,hidamount,amount):
        bots = []
        for i in range(amount):
            bot = [0,hidamount]

            if hidamount == 0:
                input = []
                for i in range(inn*out):
                    input.append(random.randint(-100000,100000)/100000)
                bot.append(input)

                input = []
                for i in range(out):
                    input.append(random.randint(-100000,100000)/100000)
                bot.append(input)
            else:
                input = []
                for i in range(inn*hid):
                    input.append(random.randint(-100000,100000)/100000)
                bot.append(input)

                input = []
                for i in range(hid):
                    
                    input.append(random.randint(-100000,100000)/100000)
                bot.append(input)
            

                for i in range(hidamount-1):
                

                    input = [] 
                    for i in range(hid*hid):
                        input.append(random.randint(-100000,100000)/100000)
                    bot.append(input)

                    input = []
                    for i in range(hid):
                        input.append(random.randint(-100000,100000)/100000)
                    bot.append(input)
                input = []
                for i in range(hid*out):
                    input.append(random.randint(-100000,100000)/100000)
                bot.append(input)

                input = []

                for i in range(out):
                    input.append(random.randint(-100000,100000)/100000)
                bot.append(input)
                bots.append(bot)

        return bots


    def botcalc(bot,inputs):
        if bot[1] == 0:
            print("")
        else:
            # set up list to not get memory studbit gFUCKING EROR
            calculationsh = []
            for i in range(len(bot[3])):
                calculationsh.append(0)
                
            
            #for input to first hidden (W)
            for i in range(len(bot[3])):
                for e in range(len(inputs)):
                    calculationsh[i] += (inputs[e] * bot[2][i + e])
                    
            
            #for input to first hidden (B)
            for i in range(len(bot[3])):
                
                calculationsh[i] += bot[3][i]

            

            #for hidden to hidden

# set up list to 
            newcalcsh = []
            
            for i in range(bot[1]-1):
                amo = 0
                amount_hid_num = 4 + (2*i)
                
                for ii in range(len(bot[3])):
                    newcalcsh.append(0)
            
                # H
                for ii in range(int(len(bot[amount_hid_num])/len(bot[amount_hid_num+1]))):
                    for e in range(len(bot[amount_hid_num+1])):
                        newcalcsh[e] += (bot[amount_hid_num][amo] * calculationsh[ii])
                        amo += 1

                for e in range(len(bot[amount_hid_num+1])):
                    newcalcsh[e] += bot[amount_hid_num+1][e]
                    
                
                calculationsh = newcalcsh


            #for hidden to output (H)

            # set up list to not get
            fcalc = []
            numoftimes = 0
            for i in range(int(len(bot[len(bot)-1]))):
                fcalc.append(0)

            #W
            
            for i in range(int(len(bot[(len(bot)-2)])/len(bot[len(bot)-1]))): # e3  
                for ii in range(int(len(bot[len(bot)-1]))): # e2        et6
                    
                    fcalc[ii] += (calculationsh[i] * bot[len(bot)-2][numoftimes])
                    numoftimes += 1

            #B
            for i in range(int(len(bot[len(bot)-1]))): # e3  
                
                fcalc[i] += bot[len(bot)-1][i]
            
            return fcalc
        
    def demonstration():
        import pygame
        fps = 25

        # Initialize Pygame
        pygame.init()
        zoom = 1
        truepos = (0*zoom,0*zoom)
        moving = 0

        innodes = 0
        hidnodes = []
        

        # Set the screen size
        screen_size = (1000,800)
        screen = pygame.display.set_mode(screen_size)

        # Set the circle parameters
        circle_distance = 125
        circle_radius = 20

        # Set the line parameters
        line_start_position = (200, 100)
        line_end_position = (200, 300)
        line_width = 10
        line_color = (0, 255, 0)

        # Draw the circle and line
        

        # Update the display
        

        # Wait for the user to quit the program
        running = True
        while running:
            #creates circles
            info =  open(r'C:\Users\rgbae\OneDrive\Desktop\HOLDER\neutralnetworkshit\Question1NN\simpleNNinfo.txt','r')
            items = []
            for line in info:
                items.append(ast.literal_eval(line.strip()))

            innodes = len(items[0])/2
            hidnodes = [((len(items)-2)/2)+1]

            #entering ALL hidden layers
            for i in range(1, len(items),2):
                for ii in range(len(items[i])):
                    hidnodes.append(items[i][ii])
            
            #CREATING INPUTS
            for i in range(int(innodes)):
                pygame.draw.circle(screen, (0,255,0), (0+truepos[0], circle_distance*i*zoom+truepos[1]), circle_radius*zoom)
            #CREATING HIDDEN
            for i in range(int(hidnodes[0])):
                
                
                for ii in range(int((len(hidnodes)-1)/hidnodes[0])):

                    
                    pygame.draw.circle(screen, (0,255,0), ( (i*circle_distance*2*zoom)+truepos[0]  , (circle_distance*ii*zoom)+truepos[1]  ) , circle_radius*zoom)

            time.sleep(1/fps)
            if moving != 0:
                truepos = (fakepos[0] + (pygame.mouse.get_pos()[0] -moving[0]), fakepos[1] + (pygame.mouse.get_pos()[1] - moving[1]))
            
            #pygame.draw.line(screen, line_color, (line_start_position[0]+truepos[0],line_start_position[1]+truepos[1]), (line_end_position[0]+truepos[0],line_end_position[1]+truepos[1]), int(line_width * zoom))
            
            for event in pygame.event.get():
                
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    
                    fakepos = truepos
                    moving = pygame.mouse.get_pos()
                    if event.button == 4:
                        
                        zoom += .025
                        truepos = (truepos[0],truepos[1])
                    if event.button == 5:
                        
                        zoom -= .025
                        truepos = (truepos[0],truepos[1])

                elif event.type == pygame.MOUSEBUTTONUP:
                    
                    moving = 0
                    if event.button == 4:
                        
                        zoom += .025
                        truepos = (truepos[0],truepos[1])
                    if event.button == 5:
                        
                        zoom -= .025
                        truepos = (truepos[0],truepos[1])
                

            
                    
            
            pygame.display.update()
            screen.fill((0,0,0))

        # Quit Pygame  [288, 1, [-0.56347, 0.38634, -0.9887, 0.78007], [0.19294, 0.79101], [-0.19267, -0.28951, 0.68305, -0.52685], [0.30016, 0.63067]]
        pygame.quit()
#START CODE 
