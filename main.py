import numpy as np
import cv2
from scipy.spatial import distance as dist
from collections import Counter

import pylab

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

# used to keep track of the contours to return
return_contours = []

input_image = cv2.imread('../test1.jpg')
input_image_hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

'''
    1. get raw color masks (cv2.inRange)
    2. use them to isolate raw colors in input image (cv2.bitwise_and)
    3. convert them to gray to ready for contours 
    4. extract contours (cv2.findConturs)
'''
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
        _, cont, _  = cv2.findContours( converted_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        colors[color]['contours'] = cont

    except KeyError:
        # Key is not present
        print('no lower/upper values for %s' % color)
        pass
    
contour_count = 0
for color in colors:
    try:
        for single_contour in colors[color]['contours']:
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
                
            return_contours.append(contour_info)
        
    except KeyError:
        #print('%s does not have any contours' % color)
        pass
        
    contour_count = 0

del contour_count, center_x, center_y, converted_mask, input_color_mask, raw_color_mask

'''
1. assume we find two bulls eye contours
2. confirm them with the current method (dist <= 3)
3. possibly augment step 2 with len(approx)>=4
4. measure distance
'''

bull_score = {}
gr_contours = [gr for gr in return_contours if gr['source']=='gr']

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
for rc in return_contours:
    for bs in bull_score:
        if bs == rc['mid'] and bull_score[bs] > 1:
            rc['confirmed'] = True
            rc['position'] = 'bull'

del gr_contours, gr_cont_check, gr_cont


# helper array 
t = [cb_mid['mid'] for cb_mid in return_contours if (cb_mid['confirmed'] and cb_mid['position']=='bull')]
avg_bull_middle = [int(sum(y) / len(y)) for y in zip(*t)] #SO: https://stackoverflow.com/a/12412686
del t

# calculate distance from cell to confirmed bull's average
for rc in return_contours:
    if not rc['position']=='bull':
        rc['distance'] =  int(dist.euclidean( avg_bull_middle , rc['mid']))

# extract every distance from non-bull contours
cell_dists = [cb['distance'] for cb in return_contours if not cb['confirmed']]


# Sort all the distances,
# since there should be only 20 of each distance
# we can determine if a distance is 'one of the 20' 
# by looking at the previous value and the next value of each distance

'''
    cell_dists.sort() = [
        0   249,
        1   250,
        2   250,                     (is next one above 4? = no , is previous one within 4? = yes  , no + yes == ok)   
        3   297, <--- problem child! (is next one above 4? = no , is previous one within 4? = no  , no + no == skip)
        4   331,                     (is next one above 4? = yes, is previous one within 4? = no  , yes + no == ok)
        5   331,
        6   332,                    (is next one above 4? = no , is previous one within 4? = yes  , no + yes == ok) 
        7   405, <--- good kid!     (is next one above 4? = yes , is previous one within 4? = no  , yes + no == ok) 
        8   406,
        9   405,
        10  407,
        ...
        82  406,        (is next one above 4? = no , is previous one within 4? = yes  , no + yes == ok) 
        83  414,        (if reached 80 'ok' values == break)            
        84  428,
        85  437,        
        86  438, 
    ]

'''


# TODO: Link cell_dists_okay back to return contours
# HINT: Sort return contours by distance first and then loop through them


# First lets get the only contours with a distance
s = [rc for rc in return_contours if not rc['position']=='bull']

import operator # move this up to imports at top
# Now we'll sort the contours based on their distance
s.sort(key=operator.itemgetter('distance'))



# Now loop through this distance sorted contour list using the +-4 iterator.
# Mark them as confirmed if they pass the test
marked_as_accepted = 0
for s_index in range(0,len(s)-1):    
    cmp_val = s[s_index]['distance']
    #print("cmp_val = %d (index = %d) " % (cmp_val, s_index))
    cmp_high_exists, cmp_low_exists = False, False
    cmp_high_ok, cmp_low_ok = False, False
    
    if s_index==0:
        cmp_val_high = s[s_index+1]['distance']
        cmp_high_exists = True
        #print("\tcmp_high = %d " % cmp_val_high)

    if s_index>0 and not s_index==len(s)-1:
        cmp_val_low = s[s_index-1]['distance']
        cmp_val_high = s[s_index+1]['distance']
        cmp_high_exists, cmp_low_exists = True, True
        #print("\tcmp_high = %d , cmp_low = %d" % (cmp_val_high, cmp_val_low))

    if s_index==len(s)-1:
        cmp_val_low = s[s_index-1]['distance']
        cmp_low_exists = True
        #print("cmp_low = %d" % (cmp_val_low))

    if cmp_high_exists:
        diff_high = cmp_val_high - cmp_val
        if(diff_high <= 4 and diff_high > -4):
            cmp_high_ok = True
            #print("\tcmp_high ok!")
    
    if cmp_low_exists:
        diff_low = cmp_val_low - cmp_val
        if(diff_low <= 4 and diff_low > -4):
            cmp_low_ok = True
            #print("\tcmp_low ok!")

    if(cmp_high_ok or cmp_low_ok):
        #print("\taccepting cmp_val!")
        s[s_index]['confirmed'] = True
        marked_as_accepted += 1
    # else:
    #     print("\tdisregarding cmp_val!")

    if(marked_as_accepted >= 80):
        #print("ending dist checking at cell_dist index %d" % s_index)
        break
    
s = [sc for sc in s if sc['confirmed']==True]

#now loop through s one last time to determine if single/inner/outer

return_contours = [rc for rc in return_contours if rc['position']=='bull']
return_contours.extend(s)

cv2.destroyAllWindows()
cv2.namedWindow('board',cv2.WINDOW_NORMAL)
cv2.resizeWindow('board', (750,750))
i = 0
for rc in return_contours:
    if not rc['position'] == 'bull':
        cv2.putText(input_image, str(rc['distance']), (rc['mid'][0]-8 , rc['mid'][1]-8),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA )
        
    # cv2.putText(input_image, str(rc['position']), (rc['mid'][0]-15 , rc['mid'][1]-15),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA )
    cv2.drawContours(input_image, [rc['contour']], -1, (153, 255, 255), 3)
    cv2.imshow('board', input_image)
    i = i + 1
    if i == len(return_contours):
        cv2.waitKey(0)

# TODO: determine cell type (single/inner/outer)
# HINT: we know distance to bull 
# dummy method:
'''
    types = [single, inner, single, outer]
    make a list of non-bull cells
    (they are sorted by distance, yes?)

    Z = 0
    loop through each contour, X
    X['type'] = types[Z]
    if( hasnext(X) && (X+1)[dist] > X[dist])
        Z++
'''


# TODO: find a way to link each cell of same type to its neighboring cell(s)
#   *   loop each contour of one cell type,
#   *   loop other contours of same type,
#   *   use ['mid'] to determine distance to other contour
#   *   (skip if dist == 0)
#   *   track each distance, and find min dist (== neighbor)
#   *   [find system/structure of how contours link up]

print('got %d confirmed cells!' % len(return_contours))
print("done")

# return triple_contours, double_contours, single_contours, bull_contours