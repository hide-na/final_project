#Python Machine Learning (Decision Tree)

#Purpose:
    Distinguish poisonous mushrooms using decision tree. Before that use entropy and finding out which explanatory variables shows the best result. 


#Method: 

    Use mushroom dataset (More information can be found on the UCI Machine Learning Repository).
    There are 22 explanatory variables and details show below. 
    Here, we pick up only 4 explanatory variables (gill_color, gill_attachment, odor, cap_color) to distinguish poisonous mushrooms.

    When we deal with variables on Decision tree, they have to be numerical type so we will transform category variables into dummy variables. 

    We will create a decision tree from the perspective of the impurity of category identification. Impurity is an index showing the state of 
    identification of whether or not it is a poisonous mushroom, and high purity means a state in which category identification cannot be performed. 
    For example, we divide data using cap_color (c =true: 1 or not c=false: 0) and then count how many poisonous mushrooms exist there.

    The Decision tree is an algorithm that distinguishes from which of the many variables the most useful conditional branch and impurity is
    used to determine the superiority or inferiority of the branch condition. Entropy is used here as an index of its impurity.
    
    formula of entropy
        reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    Entropy H 
    Where b is the base of the logarithm used. Common values of b are 2 so I will use 2. 
    P1: the rate of not poisonous mushroom
    P2: the rate of poisonous mushroom 

    When p1=p2= 0.5, entropy will be 1.0, it means when we cannot distinguish any poisonous mushrooms, entropy will be 1.
    When p1=0.001 and p2= 0.999, in other words, we can almost perfectly distinguish poisonous mushrooms, entropy will be very close to 0. 

    Information gain is the reduction in entropy or surprise by transforming a dataset and is often used in training decision trees. 
    Therefore, we use two variables (cap/_color and gill_color_b), check which variable is more useful using information gain. 

    Then, divide data using a condition which has the biggest information gain.
    After understanding the movement of decision tree, create a model of decision tree using DecisionTreeClassifier class of sklearn.tree module. 
    When we use DescriptionTreeClassifier. We set ‘entropy’ as a parameter.


#Explanatory variables 
    Attribute Information: (classes: edible=e, poisonous=p)
       1. cap-shape:                bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
       2. cap-surface:              fibrous=f, grooves=g, scaly=y, smooth=s
       3. cap-color:                brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, 
                                             red=e, white=w, yellow=y
       4. bruises?:                 bruises=t, no=f
       5. odor:                     almond=a, anise=l, creosote=c, fishy=y, foul=f,
                                    musty=m, none=n, pungent=p, spicy=s
       6. gill-attachment:          attached=a, descending=d, free=f, notched=n
       7. gill-spacing:             close=c, crowded=w, distant=d
       8. gill-size:                broad=b, narrow=n
       9. gill-color:               black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, 
                                           pink=p, purple=u, red=e, white=w, yellow=y
      10. stalk-shape:              enlarging=e, tapering=t
      11. stalk-root:               bulbous=b, club=c, cup=u, equal=e,
                                    rhizomorphs=z, rooted=r, missing=?
      12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s
      13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s
      14. stalk-color-above-ring:   brown=n, buff=b, cinnamon=c, gray=g, orange=o,
                                    pink=p, red=e, white=w, yellow=y
      15. stalk-color-below-ring:   brown=n, buff=b, cinnamon=c, gray=g, orange=o,
                                    pink=p, red=e, white=w, yellow=y
      16. veil-type:                partial=p, universal=u
      17. veil-color:               brown=n, orange=o, white=w, yellow=y
      18. ring-number:              none=n, one=o, two=t
      19. ring-type:                cobwebby=c, evanescent=e, flaring=f, large=l,
                                    none=n, pendant=p, sheathing=s, zone=z
      20. spore-print-color:        black=k, brown=n, buff=b, chocolate=h, green=r,
                                    orange=o, purple=u, white=w, yellow=y
      21. population:               abundant=a, clustered=c, numerous=n,
                                    scattered=s, several=v, solitary=y
      22. habitat:                  grasses=g, leaves=l, meadows=m, paths=p,
                                    urban=u, waste=w, woods=d

#Core packages
    The version numbers of the major Python packages that were used are listed below. 
    Please make sure that the version numbers of your installed packages are equal to, or greater than, 
    those version numbers to ensure the code examples run correctly:
        •	NumPy 1.19.5
        •	SciPy 1.6.0
        •	matplotlib 3.3.3
        •	pandas 1.2.0
        •	Seaborn 0.11.1
        •	Sklearn 
        •	Pydotplus 2.0.2 
        •	Graphviz

#Installation 
    You can install packages via the command line by entering:
    $python -m pip install –user numpy scipy matplotlib jupyter pandas seaborn sklearn pydotplus

    Homebrew has a Graphviz port.
      $ brew install graphviz




