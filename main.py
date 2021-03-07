import numpy as np
import cv2
from scipy.spatial import distance as dist
from collections import Counter

import operator # this is used to sort a list, using a list element as the sort-key
import hashlib # now we're talking...

colors = {
    # darkest is lower
    # brightest is upper
    'red': {
        'method': 'hsv',
        'lower': np.array([170,50,50]),
        'upper': np.array([180,255,255])
    },
     'green': {
        'method': 'hsv',
        'lower': np.array([60, 100, 100]),
        'upper': np.array([90, 255, 255])
    }, 
    'black': {
       'method': 'bgr',
       'lower': np.array([0, 0, 0]),
       'upper': np.array([50, 55, 55])
    },
    'white': {
       'method': 'bgr',
       'lower': np.array([ 95, 113, 136]),
       'upper': np.array([ 197, 218, 240])
    }
}

score_cell_positions = ['single_inner', 'triple', 'single_outer', 'double']


def draw_results( input_img, ret_conts ):
    cv2.destroyAllWindows()
    cv2.namedWindow('board',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('board', (750,750))
    i = 0
    for rc in ret_conts:
        if not rc['position'] == 'bull':
          cv2.putText(input_img, str(rc['position']), (rc['mid'][0]-8 , rc['mid'][1]-8),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0),2,cv2.LINE_AA )
        # cv2.putText(input_image, '*', (rc['mid'][0]-8 , rc['mid'][1]-8),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA )
        
        # cv2.putText(input_image, str(rc['position']), (rc['mid'][0]-15 , rc['mid'][1]-15),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA )
        # cv2.drawContours(input_image, [rc['contour']], -1, (153, 255, 255), 3)
        cv2.imshow('board', input_img)
        i = i + 1
        if i == len(ret_conts):
            cv2.waitKey(0)

def extract_contours(input_image):
    extracted_contours = []
    input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    for color in colors:
        print('processing %s' % color)
        try:
            # step 1
            lower = colors[color]['lower']
            upper = colors[color]['upper']

            if colors[color]['method'] == 'bgr':
                raw_color_mask = cv2.inRange(input_image, lower, upper)
            elif colors[color]['method'] == 'hsv':
                raw_color_mask = cv2.inRange(input_image_hsv, lower, upper)
            
            # step 2
            input_color_mask = cv2.bitwise_and(input_image.copy(), input_image.copy(), mask=raw_color_mask)
            
            # step 3
            converted_mask = cv2.cvtColor( input_color_mask, cv2.COLOR_BGR2GRAY )
            
            
            #step 4
            cont, _  = cv2.findContours( converted_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #_, cont, _  = cv2.findContours( converted_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            #colors[color]['contours'] = cont
            extracted_contours.append(
                {
                    'cont_data': cont,
                    'color': color
                }
            )

        except KeyError:
            # Key is not present
            print('no lower/upper values for %s' % color)
            # pass
            exit()
            #return 
    
    return extracted_contours

def contours_initialize( input_contours ):
    contour_count = 0
    ret_conts = []
    for color_contours in input_contours:
        try:
            color = color_contours['color']
            contour_data = color_contours['cont_data']
            for single_contour in contour_data:
                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(single_contour) < 50:
                    continue

                M = cv2.moments(single_contour)
                center_x = int(M['m10']/M['m00'])
                center_y = int(M['m01']/M['m00'])
                
                contour_count = contour_count + 1

                contour_info = {
                    'contour': single_contour,
                    'mid': (center_x, center_y),
                    'source': 'bw',
                    'confirmed': False,
                    'position': 'unknown'
                }

                if color == 'green' or color=='red':
                    contour_info['source'] = 'gr'
                    
                ret_conts.append(contour_info)
            
        except KeyError:
            print('%s does not have any contours' % color)
            pass
        
    contour_count = 0
    return ret_conts

def determine_bull( input_contours ):
    bull_score = {}
    gr_contours = [gr for gr in input_contours if gr['source']=='gr']

    # loop through every bull cell and add them to a score dict
    for gr_cont in gr_contours:
        #x = gr_cont['mid']
        for gr_cont_check in gr_contours:
            if gr_cont['mid'] == gr_cont_check['mid']:
                continue

            D = dist.euclidean(gr_cont['mid'], gr_cont_check['mid'])
            if D <= 3:
                try:
                    score = bull_score[gr_cont['mid']]
                    bull_score[gr_cont['mid']] = score + 1
                    #print('increased score for bulls at %d, %d to score=%d' % (x[0], x[1], score+1) )
                except KeyError:
                    bull_score[gr_cont['mid']] = 1
                    #print('adding score to bulls at %d, %d' % (x[0], x[1] ) )
                    pass

        # if a bull point got over 1 in score, set 'confirmed' to True
        for rc in input_contours:
            for bs in bull_score:
                if bs == rc['mid'] and bull_score[bs] > 1:
                    rc['confirmed'] = True
                    rc['position'] = 'bull'

    return input_contours

def determine_bull_distances( input_contours ):
    
    # extract all x-y location of bull's mid-point
    t = [cb_mid['mid'] for cb_mid in input_contours if (cb_mid['confirmed'] and cb_mid['position']=='bull')]
    
    # from all these (x, y) values, calculate the average (x, y) value
    avg_bull_middle = [int(sum(y) / len(y)) for y in zip(*t)] #SO: https://stackoverflow.com/a/12412686

    # calculate distance from cell to confirmed bull's average
    for ic in input_contours:
        if not ic['position']=='bull':
            ic['distance'] =  int(dist.euclidean( avg_bull_middle , ic['mid']))

    return input_contours
    
def confirm_cells ( input_cell_contours ):
    # Since there should be only 20 of each distance, we can determine
    #   if a distance is 'one of the 20' by looking at the
    #   previous value and the next value of each distance.

    '''
        cell_dists.sort() = [
            1   250,
            2   250,                     (is next one above 4? = no , is previous one within 4? = yes  , no + yes == ok)   
            3   297, <--- problem child! (is next one above 4? = no , is previous one within 4? = no  , no + no == skip)
            4   331,                     (is next one above 4? = yes, is previous one within 4? = no  , yes + no == ok)
            5   331,
            6   332,                    (is next one above 4? = no , is previous one within 4? = yes  , no + yes == ok) 
            7   405, <--- good kid!     (is next one above 4? = yes , is previous one within 4? = no  , yes + no == ok) 
            8   405,
            9   407,
            ...
            82  406,        (is next one above 4? = no , is previous one within 4? = yes  , no + yes == ok) 
            83  414,        (if reached 80 'ok' values == break)            
            84  428,
            86  438, 
        ]

    '''
    # First lets get the only cell contours
    cell_contours = [rc for rc in input_cell_contours if not rc['position']=='bull']

    # Now sort these contours based on their distance
    cell_contours.sort(key=operator.itemgetter('distance'))
    
    # Now loop through this distance sorted contour list using +-4 test.
    # Mark them as confirmed if they pass the test.
    #
    # That is, if the distance from a cell to the next and/or previous cell is more than ±4,
    #   then this is not considered a 'confirmed' cell.
    # The goal is to remove non-point-cells (i.e. if contour is off a letter, number, etc.)
    accepted_contours = 0
    # during this loop, we need to check if there's a
    for contour_index in range(0,len(cell_contours)-1):    
        contour_dist = cell_contours[contour_index]['distance']
        #print('cmp_val = %d (index = %d) ' % (cmp_val, s_index))
        next_exists, prev_exists = False, False
        next_ok, prev_ok = False, False
        
        # we're at the start of the list, so we only have the next cell
        if contour_index==0:
            cont_dist_next = cell_contours[contour_index+1]['distance']
            next_exists = True
            #print('\tcmp_high = %d ' % cmp_val_high)

        # we're somewhere between the start and end of the list, so both next & previous cells exist
        if contour_index>0 and not contour_index==len(cell_contours)-1:
            prev_cont_dist = cell_contours[contour_index-1]['distance']
            cont_dist_next = cell_contours[contour_index+1]['distance']
            next_exists, prev_exists = True, True

        # we are at the end of the list, so only the previous cell exists
        if contour_index==len(cell_contours)-1:
            prev_cont_dist = cell_contours[contour_index-1]['distance']
            prev_exists = True

        if next_exists:
            dist_to_next = cont_dist_next - contour_dist
            if(dist_to_next <= 4 and dist_to_next > -4):
                next_ok = True
        
        if prev_exists:
            dist_to_prev = prev_cont_dist - contour_dist
            if(dist_to_prev <= 4 and dist_to_prev > -4):
                prev_ok = True

        if(next_ok or prev_ok):
            cell_contours[contour_index]['confirmed'] = True
            accepted_contours += 1

        # 20 [pts] * 4 [cells pr point] = 80
        if(accepted_contours >= 80):
            #print('ending dist checking at cell_dist index %d' % s_index)
            break
     
     # from now on, we're only interested in the confirmed contours
    confirmed_contours = [cc for cc in cell_contours if cc['confirmed'] == True]
    
    return confirmed_contours

def assign_cell_type( input_cell_contours ):
    # Now determine the cell types.
    # For the following to work, the contours must be sorted by their bull distance in ascending order.
    cont_type_index = 0
    for conf_con_index in range(0,len(input_cell_contours)):
        # if we're at the last contour, we know that the type is the last 'position'
        if conf_con_index == len(input_cell_contours)-1:
            input_cell_contours[conf_con_index]['position'] = score_cell_positions[len(score_cell_positions)-1]
            break
            
        # otherwise, we simply take the current and next contour's distances and diff them
        cont_dist_curr = input_cell_contours[conf_con_index]['distance']
        cont_dist_next = input_cell_contours[conf_con_index+1]['distance']
        # if we're more than 20 from the next contour, we must be advancing to next 'position'
        if cont_dist_next - cont_dist_curr > 20:
            print(f'advancing type index, diff is {cont_dist_next - cont_dist_curr}')
            cont_type_index += 1
            
        
        # and finally assign the current 'position' 
        input_cell_contours[conf_con_index]['position'] = score_cell_positions[cont_type_index]
        

    return input_cell_contours

# in order to pair them up later, we need to assign unique but determinable ID's to each contour
def generate_cont_id(contour_data, contour_distance, contour_position):
    a = str(np.amax(contour_data)) + str(np.amin(contour_data)) + str(contour_distance) + contour_position
    return hashlib.md5(a.encode()).hexdigest()

def assign_cell_id( input_cell_contours ):
    # we keep track of the assigned id's 
    assigned_ids = {}
    for icc in input_cell_contours:
        uniq_id = generate_cont_id(
            icc['contour'],
            icc['distance'],
            icc['position']
        )

    # if we ever get an already assigned id, just generate a new one from the current (but unassigned) id
    if uniq_id in assigned_ids:
        print('!!!Duplicate ID!!!')
        uniq_id = hashlib.md5(uniq_id.encode()).hexdigest()

    icc['id'] = uniq_id
    assigned_ids[uniq_id] = uniq_id

    return input_cell_contours

def determine_linked_cells ( input_cell_contours ):
    for pos in score_cell_positions:
        curr_pos = [contours for contours in input_cell_contours if contours['position']==pos]
        print(f'looping {pos} , L = {len(curr_pos)}')


    return input_cell_contours

def main( image_path ):
    return_contours = []
    #input_image = cv2.imread('../test1.jpg')
    input_image = cv2.imread('./images/test2.jpg')

    # extract contours under each different color
    extracted_contours = extract_contours( input_image )

    # from each color's contours, we initialize the return datatypes
    return_contours = contours_initialize( extracted_contours )
    
    # from these contours, we now try to determine which one of them is the bull's eye
    return_contours = determine_bull( return_contours )

    # determine each contours' mid distance to bull
    return_contours = determine_bull_distances( return_contours )
    

    # now we'll work a bit on these score cells, so split up the contours for now
    score_cell_contours = [rc for rc in return_contours if not rc['position']=='bull']
    remaining_contours = [rc for rc in return_contours if rc['position']=='bull']

    score_cell_contours = confirm_cells ( score_cell_contours )
    score_cell_contours = assign_cell_type( score_cell_contours )
    score_cell_contours = assign_cell_id ( score_cell_contours )
    score_cell_contours = determine_linked_cells ( score_cell_contours ) 

    # now recombine them
    remaining_contours.extend(score_cell_contours)
    return_contours = remaining_contours


    print('got %d confirmed cells!' % len(return_contours))
    print('done')

    draw_results( input_image , return_contours ) 
    return return_contours

x = main("")