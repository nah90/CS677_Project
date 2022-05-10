"""
Nathan A. Horak
Class: CS 677 - Spring 02
Date: 4/27/2021
Class Project
Description: Perform data analysis on a dataset containing match results from games played
on the Free Internet Chess Server in 2016. Examine effectiveness and use of openers, predicting
whether a human player was in the match, and match outcome.
"""
import warnings
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Constants
INPUT_DIR = r'C:\Users\Nathan\Documents\Grad School\Python Scripts\MET CS 677\project'
INPUT_ONE = '2016_CvC.csv'  # grab CvC and CvH data
INPUT_TWO = '2016_CvH.csv'
INPUT_FILE_ONE = os.path.join(INPUT_DIR, INPUT_ONE)  # input
INPUT_FILE_TWO = os.path.join(INPUT_DIR, INPUT_TWO)
COL_LIST = ['White Elo', 'Black Elo', 'WhiteIsComp', 'BlackIsComp', 'Result-Winner',
            'Moves']
MOVE_LIST = ['White One', 'Black One', 'White Two', 'Black Two', 'White Three', 'Black Three']
PRED_DICT = {'White Elo': 'White Elo', 'Black Elo': 'Black Elo',
             'open_labels': 'open_labels', 'winner_labels': 'winner_labels', 'human_labels': 'human_labels',
             'whiteopen_labels': 'whiteopen_labels', 'blackopen_labels': 'blackopen_labels'}
OPENER = ['Opener']
ROUND = 1  # round
PERC_DATA = 0.01  # percent of dataset to sample
MAX_CHESS_OPEN_LEN = 5  # max number of chess moves in opener
START_K = 5  # starting K for kNN
END_K = 13  # end
STEP_K = 2
N_MIN = 2  # number of estimators for Random Forest
N_MAX = 10
STEP_N = 1
D_MIN = 2  # depth of Random Forest tree
D_MAX = 6
STEP_D = 1
BAR_WIDTH = 0.25  # bar width for plots

# Chess Openings
RUY_LOPEZ = ['e4', 'e5', 'Nf3', 'Nc6', 'Bb5']  # chess openers in list of str
ITALIAN_GAME = ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4']
SICILIAN_DEF = ['e4', 'c5']
FRENCH_DEF = ['e4', 'e6']
CARO_KANN_DEF = ['e4', 'c6']
PIRC_DEF = ['e4', 'd6']
QUEENS_GAMBIT = ['d4', 'd5', 'c4']
INDIAN_DEF = ['d4', 'Nf6']
ENGLISH_OPEN = ['c4']
RETI_OPEN = ['Nf3']

# Dicts
OPENING_LABELS_DICT = {'No Opener': 0, 'Ruy Lopez': 1, 'Italian Game': 2, 'Queens Gambit': 3,
                       'English Open': 4, 'Reti Open': 5, 'Sicilian Defense': 6, 'French Defense': 7,
                       'Caro Kann Defense': 8,
                       'Pirc Defense': 9, 'Indian Defense': 10}  # dicts
WINNER_LABELS_DICT = {'Draw': 0, 'White': 1, 'Black': 2}
HUMAN_LABELS_DICT = {'Computer vs Computer': 0, 'Human on White': 1, 'Human on Black': 2}
WHITEOPEN_LABELS_DICT = {'e4': 0, 'd4': 1, 'c4': 2, 'Nf3': 3}
BLACKOPEN_LABELS_DICT = {'e5': 0, 'c5': 1, 'e6': 2, 'c6': 3, 'd6': 4}

# Openers by Player
WLAST_OPEN = [RUY_LOPEZ, ITALIAN_GAME, QUEENS_GAMBIT, ENGLISH_OPEN, RETI_OPEN]
# openers with white having last move
BLAST_OPEN = [SICILIAN_DEF, FRENCH_DEF, CARO_KANN_DEF, PIRC_DEF, INDIAN_DEF]
# openers with black having last move
ALL_OPEN = WLAST_OPEN + BLAST_OPEN


# Functions
def chess_open_percent(winner: str, chess_opening: list, df: pd.DataFrame()):
    """
    Given a string of a match outcome, a list of strings as the opener, and a dataframe give a
    match outcome percentage for the input string

    Parameters
    ----------
    winner : str
        String of match outcome
    chess_opening : list
        List of str of chess_opening
    df : pd.DataFrame()
        Dataframe to analyze chess matches from

    Returns
    -------
    perc : float
        Percentage of match outcome

    """
    move_df = df.copy()  # copy df
    move_df['Check'] = 0  # new column
    for e in range(len(move_df)):  # loop through df
        valid = True  # check starts at 'True'
        for f in range(len(chess_opening)):  # match each
            if chess_opening[f] == move_df.iloc[e][f + 4]:  # matches list with df column
                pass
            else:
                valid = False  # if doesn't match, invalidate and break
                break
        if valid == True:
            move_df['Check'][e] = 1  # if valid, check is 1
    move_df = move_df[move_df.Check != 0]  # move
    move_df = move_df.drop(columns='Check')  # drop check
    try:  # try block for win%
        perc = ((move_df.value_counts(subset='Result-Winner')).loc[winner] / len(move_df) * 100).round(ROUND)
    except KeyError:
        perc = 0  # if 0 error, set percent to 0
    return perc  # return


def outcome_open_list(white_list: list, black_list: list, draw_list: list, chess_opening: list, df: pd.DataFrame()):
    """
    Given a list of outcomes for white win, black win, draws as well a list of strings as the opener
    and the dataframe return updated white win, black win, and draw lists

    Parameters
    ----------
    white_list : list
        List of match outcome percentages for white
    black_list : list
        List of match outcome percentages for black
    draw_list : list
        List of match outcome percentages for a draw
    chess_opening : list
        List of str of chess_opening
    df : pd.DataFrame()
        Dataframe to analyze chess matches from

    Returns
    -------
    white_list : list
        List of match outcome percentages for white
    black_list : list
        List of match outcome percentages for black
    draw_list : list
        List of match outcome percentages for a draw

    """
    white_list.append(chess_open_percent('White', chess_opening, df))  # add outcome to white list
    black_list.append(chess_open_percent('Black', chess_opening, df))  # to black
    draw_list.append(chess_open_percent('Draw', chess_opening, df))  # to draw
    return white_list, black_list, draw_list


def non_open_percent(assign_opens: bool, open_list_of_list: list, df: pd.DataFrame()):
    """
    

    Parameters
    ----------
    assign_opens : bool
        Boolean of whether or not to assign openers
    open_list_of_list : list
        List of list of strings of openers
    df : pd.DataFrame()
        Dataframe to analyze chess matches from

    Returns
    -------
    move_df : pd.DataFrame()
        DataFrame of all matches not having openers in the input list of list

    """
    move_df = df.copy()  # copy df
    move_df['Drop'] = 0  # create drop column
    for e in range(len(move_df)):  # loop through df
        valid = True  # check starts at 'True'
        for f in range(len(open_list_of_list)):  # loop through list of openers
            move_list = []  # create blank list
            move_str = ''  # blank str
            for g in range(len(open_list_of_list[f])):  # loop through each opener list
                if move_df.iloc[e][g + 4] != 0:  # exit if '0' means disconnect/left
                    move_list.append(move_df.iloc[e][g + 4])  # append
                else:
                    continue
            move_str = ''.join(move_list)  # to str
            if move_str == ''.join(open_list_of_list[f]):  # if matches list of opener
                valid = False  # its an opener, false
                if assign_opens is True:
                    df['Opener'][e] = f + 1  # if boolean is true, assign number associated
            if valid == False:
                break  # break on false
            else:
                pass
        if valid == False:
            pass  # pass on false
        else:
            move_df['Drop'][e] = 1  # if df moves match an opener
    move_df = move_df[move_df.Drop != 0]  # drop
    move_df = move_df.drop(columns='Drop')
    return move_df  # return


def knn_ml(class_pred: str, class_dict: dict, START_K: int, END_K: int, STEP_K: int, df: pd.DataFrame()):
    """
    Given a string to predict, dictionary of attributes, starting 'k' value, ending 'k' value, step between each 'k' value, and
    input dataframe calculates the optimal 'k*' value and returns the predicted values,
    actual values, the optimal k* value, and the accuracy of the optimal k* value
    Parameters
    ----------
    class_pred : str
        String of attribute to predict
    class_dict : dict
        Dict of strings for attributes/predict
    START_K : int
        The starting 'k' value
    END_K : int
        The ending 'k' value
    STEP_K : int
        Step between each 'k' value
    df : pd.DataFrame()
        DataFrame to analyze

    Returns
    -------
    opt_pred : np.array
        An array of predicted values for testing portion
    opt_test : pd.Series
        Series of actual test values
    opt_k : TYPE
        String of optimal 'k' value
    opt_measure : float
        Float of percentage accuracy in predicting 'Y' Test values

    """
    ndf = df.copy()  # copy df
    ndict = class_dict.copy()  # copy class dict
    ndf['open_labels'] = ndf['Opener'].map(OPENING_LABELS_DICT)  # assign via map dict as dummmies
    ndf['winner_labels'] = ndf['Result-Winner'].map(WINNER_LABELS_DICT)
    ndf['human_labels'] = ndf['Human Player'].map(HUMAN_LABELS_DICT)
    ndf['whiteopen_labels'] = ndf['White One'].map(WHITEOPEN_LABELS_DICT)
    ndf.loc[ndf['whiteopen_labels'].isna(), 'whiteopen_labels'] = 0
    ndf['blackopen_labels'] = ndf['Black One'].map(BLACKOPEN_LABELS_DICT)
    ndf.loc[ndf['blackopen_labels'].isna(), 'blackopen_labels'] = 0
    y = ndict.pop(class_pred)  # pop out of dict the classifier to predict, assign to y
    Y = ndf[[y]].values  # assign Y to df values
    ndf = ndf.drop(columns=class_pred)  # remove Y from df
    x_list = list(ndict.keys())  # make list of remaining keys in ndict
    X = ndf[x_list].values  # assign X to values
    opt_pred = []  # blank list
    opt_k = ''  # for optimal
    opt_measure = 0
    print("Using attributes: ", x_list, ", predict: ", [y])  # print attributes used and predicted
    for f in range(START_K, END_K + 1, STEP_K):  # loop through each 'k' value
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y)
        # split into train/test for X/Y with stratify
        scaler = StandardScaler().fit(X)  # fit scasler
        X_train = scaler.transform(X_train)  # transform
        X_test = scaler.transform(X_test)
        knn_class = KNeighborsClassifier(n_neighbors=f)
        knn_class.fit(X_train, Y_train.ravel())  # fit
        pred = knn_class.predict(X_test)
        if (1 - np.mean(pred != Y_test.ravel())) > opt_measure:  # set optimal
            opt_pred = pred  # assign if optimal
            opt_test = Y_test
            opt_k = 'k=' + str(f)
            opt_measure = (1 - np.mean(pred != Y_test.ravel()))  # accuracy
    return opt_pred, opt_test, opt_k, (opt_measure * 100).round(ROUND)  # returns


def nbt_ml(class_pred: str, class_dict: dict, df: pd.DataFrame()):
    """
    

    Parameters
    ----------
    class_pred : str
        String of attribute to predict
    class_dict : dict
        Dict of strings for attributes/predict
    df : pd.DataFrame()
        DataFrame to analyze

    Returns
    -------
    Y_pred : np.array
        Array of prediction values
    Y_test : pd.Series
        Series of actual test values
    nbt_measure : float
        Float of percentage accuracy in predicting 'Y' Test values

    """
    ndf = df.copy()  # copy df
    ndict = class_dict.copy()  # copy class dict
    ndf['open_labels'] = ndf['Opener'].map(OPENING_LABELS_DICT)  # assign via map dict as dummmies
    ndf['winner_labels'] = ndf['Result-Winner'].map(WINNER_LABELS_DICT)
    ndf['human_labels'] = ndf['Human Player'].map(HUMAN_LABELS_DICT)
    ndf['whiteopen_labels'] = ndf['White One'].map(WHITEOPEN_LABELS_DICT)
    ndf.loc[ndf['whiteopen_labels'].isna(), 'whiteopen_labels'] = 0
    ndf['blackopen_labels'] = ndf['Black One'].map(BLACKOPEN_LABELS_DICT)
    ndf.loc[ndf['blackopen_labels'].isna(), 'blackopen_labels'] = 0
    y = ndict.pop(class_pred)  # pop out of dict the classifier to predict, assign to y
    Y = ndf[[y]].values  # assign Y to df values
    ndf = ndf.drop(columns=class_pred)  # remove Y from df
    x_list = list(ndict.keys())  # make list of remaining keys in ndict
    X = ndf[x_list].values  # assign X to values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y)
    # split into train/test for X/Y with stratify
    print("Using attributes: ", x_list, ", predict: ", [y])  # print attributes used and predicted
    nbt_class = MultinomialNB().fit(X_train, Y_train.ravel())  # fit NB
    Y_pred = nbt_class.predict(X_test)  # predict
    nbt_measure = ((1 - np.mean(Y_pred != Y_test.ravel())) * 100).round(ROUND)  # accuracy
    return Y_pred, Y_test, nbt_measure  # return


def dt_ml(class_pred: str, class_dict: dict, df: pd.DataFrame()):
    """
    

    Parameters
    ----------
    class_pred : str
        String of attribute to predict
    class_dict : dict
        Dict of strings for attributes/predict
    df : pd.DataFrame()
        DataFrame to analyze

    Returns
    -------
    Y_pred : np.array
        Array of prediction values
    Y_test : pd.Series
        Series of actual test values
    dt_measure : float
        Float of percentage accuracy in predicting 'Y' Test values

    """
    ndf = df.copy()  # copy df
    ndict = class_dict.copy()  # copy class dict
    ndf['open_labels'] = ndf['Opener'].map(OPENING_LABELS_DICT)  # assign via map dict as dummmies
    ndf['winner_labels'] = ndf['Result-Winner'].map(WINNER_LABELS_DICT)
    ndf['human_labels'] = ndf['Human Player'].map(HUMAN_LABELS_DICT)
    ndf['whiteopen_labels'] = ndf['White One'].map(WHITEOPEN_LABELS_DICT)
    ndf.loc[ndf['whiteopen_labels'].isna(), 'whiteopen_labels'] = 0
    ndf['blackopen_labels'] = ndf['Black One'].map(BLACKOPEN_LABELS_DICT)
    ndf.loc[ndf['blackopen_labels'].isna(), 'blackopen_labels'] = 0
    y = ndict.pop(class_pred)  # pop out of dict the classifier to predict, assign to y
    Y = ndf[[y]].values  # assign Y to df values
    ndf = ndf.drop(columns=class_pred)  # remove Y from df
    x_list = list(ndict.keys())  # make list of remaining keys in ndict
    X = ndf[x_list].values  # assign X to values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y)
    # split into train/test for X/Y with stratify
    print("Using attributes: ", x_list, ", predict: ", [y])  # print attributes used and predicted
    dt_class = tree.DecisionTreeClassifier(criterion='entropy')  # dt
    dt_class = dt_class.fit(X_train, Y_train)  # fit
    Y_pred = dt_class.predict(X_test)  # predict
    dt_measure = ((1 - np.mean(Y_pred != Y_test.ravel())) * 100).round(ROUND)  # accuracy
    return Y_pred, Y_test, dt_measure  # return


def rf_ml(class_pred: str, class_dict: dict, df: pd.DataFrame()):
    """
    

    Parameters
    ----------
    class_pred : str
        String of attribute to predict
    class_dict : dict
        Dict of strings for attributes/predict
    df : pd.DataFrame()
        DataFrame to analyze

    Returns
    -------
    opt_pred : np.array
        An array of predicted values for testing portion
    opt_test : pd.Series
        Series of actual test values
    opt_rf : TYPE
        String of optimal 'n' and 'd' values
    opt_measure : float
        Float of percentage accuracy in predicting 'Y' Test values

    """
    ndf = df.copy()  # copy df
    ndict = class_dict.copy()  # copy class dict
    ndf['open_labels'] = ndf['Opener'].map(OPENING_LABELS_DICT)  # assign via map dict as dummmies
    ndf['winner_labels'] = ndf['Result-Winner'].map(WINNER_LABELS_DICT)
    ndf['human_labels'] = ndf['Human Player'].map(HUMAN_LABELS_DICT)
    ndf['whiteopen_labels'] = ndf['White One'].map(WHITEOPEN_LABELS_DICT)
    ndf.loc[ndf['whiteopen_labels'].isna(), 'whiteopen_labels'] = 0
    ndf['blackopen_labels'] = ndf['Black One'].map(BLACKOPEN_LABELS_DICT)
    ndf.loc[ndf['blackopen_labels'].isna(), 'blackopen_labels'] = 0
    y = ndict.pop(class_pred)  # pop out of dict the classifier to predict, assign to y
    Y = ndf[[y]].values  # assign Y to df values
    ndf = ndf.drop(columns=class_pred)  # remove Y from df
    x_list = list(ndict.keys())  # make list of remaining keys in ndict
    X = ndf[x_list].values  # assign X to values
    opt_pred = []  # blank list
    opt_n_int = 0  # opt n int
    opt_d_int = 0  # opt d int
    opt_measure = 0  # opt measure
    print("Using attributes: ", x_list, ", predict: ", [y])
    for e in range(N_MIN, N_MAX + 1, STEP_N):  # loop for n
        for f in range(D_MIN, D_MAX + 1, STEP_D):  # loop for d
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y)
            # split into train/test for X/Y with stratify
            rf_class = RandomForestClassifier(n_estimators=e, max_depth=f, criterion='entropy')
            rf_class = rf_class.fit(X_train, Y_train.ravel())  # fit
            Y_pred = rf_class.predict(X_test)  # predict
            if (1 - np.mean(Y_pred != Y_test.ravel())) > opt_measure:  # set optimal
                opt_pred = Y_pred  # assign if optimal
                opt_test = Y_test
                opt_n_int = e
                opt_d_int = f
                opt_measure = (1 - np.mean(Y_pred != Y_test.ravel()))  # opt measure
    opt_rf = 'n= ' + str(opt_n_int) + ', d= ' + str(opt_d_int)  # opt rf n,d
    return opt_pred, opt_test, opt_rf, ((opt_measure * 100).round(ROUND))  # return


# Pandas Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 50)
pd.options.mode.chained_assignment = None

# Setup
if os.path.exists(INPUT_FILE_ONE):
    cvc_df = pd.read_csv(INPUT_FILE_ONE, header=0, usecols=COL_LIST)
else:
    assert False, f"The file '{INPUT_FILE_ONE}' does not exist!"  # asserts

if os.path.exists(INPUT_FILE_TWO):
    cvh_df = pd.read_csv(INPUT_FILE_TWO, header=0, usecols=COL_LIST)
else:
    assert False, f"The file '{INPUT_FILE_TWO}' does not exist!"  # asserts

chess_df = cvc_df.append(cvh_df, ignore_index=True)  # combine CvC and CvH df
chess_df = chess_df.sample(frac=PERC_DATA)  # sample
chess_df = chess_df.reset_index(drop=True)
chess_df['Human Player'] = 'Computer vs Computer'  # df formatting
chess_df['White One'] = 0  # assign new columns for first six moves of game
chess_df['Black One'] = 0
chess_df['White Two'] = 0
chess_df['Black Two'] = 0
chess_df['White Three'] = 0
chess_df['Black Three'] = 0
chess_df.loc[chess_df['WhiteIsComp'] == 'No', 'Human Player'] = 'Human on White'  # assign human on black/white
chess_df.loc[chess_df['BlackIsComp'] == 'No', 'Human Player'] = 'Human on Black'
chess_df = chess_df.drop(columns='WhiteIsComp')  # drop old columns
chess_df = chess_df.drop(columns='BlackIsComp')

for e in range(len(chess_df)):  # loop through df
    turn_list = []
    chess_iter = chess_df['Moves'][e].split(' ')  # iterate through 'Moves' column
    chess_iter[:] = [x for x in chess_iter if '.' not in x]  # ignore . in iteration
    chess_iter = [s for s in chess_iter if s != ""]  # ignore blanks in iteration
    for f in range(MAX_CHESS_OPEN_LEN + 1):  # loop through number equal to max number of moves in opener
        if f < len(chess_iter):
            turn_list.append(chess_iter[f])
        else:
            turn_list.append(0)
    for g in range(MAX_CHESS_OPEN_LEN + 1):  # add to appropriate column
        chess_df.iloc[e, [g + 5]] = turn_list[g]

chess_df = chess_df.drop(columns='Moves')  # drop 'Moves' column
chess_df['Opener'] = 0  # create 'Opener' column
wlast_df = non_open_percent(False, WLAST_OPEN, chess_df)
# get df of all games with no white last move opener
blast_df = non_open_percent(False, BLAST_OPEN, chess_df)
# get df of all games with no black last move opener
no_open_df = non_open_percent(True, ALL_OPEN, chess_df)
# create df of non-openers and assign opener labels

open_dict = {v: k for k, v in OPENING_LABELS_DICT.items()}  # create dict of openers
chess_df['Opener'] = chess_df['Opener'].map(open_dict)  # map to df column 'Opener'

# Opening Analysis
white_fig = plt.figure()  # analyze win rates of white openers and all others
ax = white_fig.add_axes([0, 0, 1, 1])
white_opens = ['Ruy Lopez', 'Italian', 'Queens Gambit', 'English Open', 'Reti Open', 'Other']
wlist = []
blist = []
dlist = []
white_win_perc = [wlist, blist, dlist]  # list of list for white win percentage
for e in range(len(WLAST_OPEN)):  # loop through white openers and call to function, append list
    wlist, blist, dlist = outcome_open_list(wlist, blist, dlist, WLAST_OPEN[e], chess_df)
# call 'outcome_open_list' function
wlist.append(((wlast_df.value_counts(subset='Result-Winner')).loc['White'] / len(wlast_df) * 100).round(ROUND))
blist.append(((wlast_df.value_counts(subset='Result-Winner')).loc['Black'] / len(wlast_df) * 100).round(ROUND))
dlist.append(((wlast_df.value_counts(subset='Result-Winner')).loc['Draw'] / len(wlast_df) * 100).round(ROUND))
# do same for all other openers
ax.set_ylabel('Percentage of Match Outcome')  # graph titles
ax.set_title('Outcome Breakdown for White Openers and Other')
ax.legend(labels=['White', 'Black', 'Draw'])
plt.xticks([p + BAR_WIDTH for p in range(len(white_win_perc[0]))], white_opens)  # bar chart
ax.bar(np.arange(len(white_opens)) + 0.00, white_win_perc[0], color='b', width=BAR_WIDTH, label='White')
ax.bar(np.arange(len(white_opens)) + 0.25, white_win_perc[1], color='r', width=BAR_WIDTH, label='Black')
ax.bar(np.arange(len(white_opens)) + 0.50, white_win_perc[2], color='g', width=BAR_WIDTH, label='Draw')
# bar chart setup
plt.legend()
plt.show()

black_fig = plt.figure()  # analyze win rates of black openers and all others
ax = black_fig.add_axes([0, 0, 1, 1])
black_opens = ['Sicilian Def.', 'French Def.', 'Caro Kann Def.', 'Pirc Def.', 'Indian Def.', 'Other']
wlist = []
blist = []
dlist = []
black_win_perc = [wlist, blist, dlist]  # list of list for black win percentage
for e in range(len(BLAST_OPEN)):  # loop through black openers and call to function, append list
    wlist, blist, dlist = outcome_open_list(wlist, blist, dlist, BLAST_OPEN[e], chess_df)
    # call 'outcome_open_list' function
wlist.append(((blast_df.value_counts(subset='Result-Winner')).loc['White'] / len(blast_df) * 100).round(ROUND))
blist.append(((blast_df.value_counts(subset='Result-Winner')).loc['Black'] / len(blast_df) * 100).round(ROUND))
dlist.append(((blast_df.value_counts(subset='Result-Winner')).loc['Draw'] / len(blast_df) * 100).round(ROUND))
# do same for all other openers
ax.set_ylabel('Percentage of Match Outcome')  # graph titles
ax.set_title('Outcome Breakdown for Black Openers and Other')
ax.legend(labels=['White', 'Black', 'Draw'])
plt.xticks([p + BAR_WIDTH for p in range(len(black_win_perc[0]))], black_opens)  # bar chart
ax.bar(np.arange(len(black_opens)) + 0.00, black_win_perc[0], color='b', width=BAR_WIDTH, label='White')
ax.bar(np.arange(len(black_opens)) + 0.25, black_win_perc[1], color='r', width=BAR_WIDTH, label='Black')
ax.bar(np.arange(len(black_opens)) + 0.50, black_win_perc[2], color='g', width=BAR_WIDTH, label='Draw')
# bar chart setup
plt.legend()
plt.show()

indianchk_df = chess_df.loc[(chess_df['White One'] == 'd4')]  # create df for indian defense
indianchk_df.loc[indianchk_df['Opener'] == 'Queens Gambit', 'Opener'] = 'No Opener'
# assign queens gambit games as other

ruyvsitaly_df = chess_df.loc[(chess_df['White One'] == 'e4') & (chess_df['Black One'] == 'e5') &
                             (chess_df['White Two'] == 'Nf3') & (chess_df['Black Two'] == 'Nc6')]
# create df to analyze ruy vs italian

e4open_df = chess_df.loc[(chess_df['White One'] == 'e4')]  # create df for black response to 'W1: e4'
e4open_df.loc[e4open_df['Opener'] == 'Ruy Lopez', 'Opener'] = 'No Opener'
e4open_df.loc[e4open_df['Opener'] == 'Italian Game', 'Opener'] = 'No Opener'
# want to examine all four black "defense" - include any Ruy/Italian as 'other'

queens_df = chess_df.loc[(chess_df['White One'] == 'd4') & (chess_df['Black One'] == 'd5')]
# create df for queens gambit

chess_dict = PRED_DICT.copy()  # copy dict from constant
full_dict = chess_dict.copy()  # create full dict
full_dict.pop('whiteopen_labels', None)
full_dict.pop('blackopen_labels', None)
# used to predict from Black ELO, White ELO, Open Labels, Winner Labels, and Human Labels

whitethird_dict = chess_dict.copy()
whitethird_dict.pop('open_labels', None)
whitethird_dict.pop('winner_labels', None)
whitethird_dict.pop('blackopen_labels', None)
# used to predict White Opener with attributes Black ELO, White ELO, and Human Labels

blackthird_dict = chess_dict.copy()
blackthird_dict.pop('open_labels', None)
blackthird_dict.pop('winner_labels', None)
# used to predict Black Opener with attributes Black ELO, White ELO, White Opener, and Human Labels

third_dict = chess_dict.copy()  # copy dict from chess_dict
third_dict.pop('whiteopen_labels', None)  # used to predict
third_dict.pop('blackopen_labels', None)
third_dict.pop('winner_labels', None)
# used to predict from Black ELO, White ELO, Open Labels, and Human Labels


# Opening
print("")
print("Machine Learning & Data Science Analysis on Chess Openers from 2016 Online Chess Data Set")
print("")
print(f"Analysis done using {PERC_DATA * 100}% of overall data set, sampled randomly.")
print("")  # intro with % sample
# kNN
print("")
print("")
print("kNN Analysis-")
print("")
print("")  # kNN
print("")
print("Predicting Human vs Computer, Match Outcome, Chess Openers, White First Move, and Black First Move-")
print("")  # predict prints
knn_humanvscomp_pred, knn_humanvscomp_test, knn_humanvscomp_k, knn_humanvscomp_measure = knn_ml('human_labels',
                                                                                                full_dict, START_K,
                                                                                                END_K, STEP_K, chess_df)
print("")
print(
    f"Using kNN, the k* value '{knn_humanvscomp_k}' was the most accurate with an accuracy of {knn_humanvscomp_measure}% for predicting human players.")
print("")
print("")  # predict prints
knn_fulloutcome_pred, knn_fulloutcome_test, knn_fulloutcome_k, knn_fulloutcome_measure = knn_ml('winner_labels',
                                                                                                full_dict, START_K,
                                                                                                END_K, STEP_K, chess_df)
print("")
print(
    f"Using kNN, the k* value '{knn_fulloutcome_k}' was the most accurate with an accuracy of {knn_fulloutcome_measure}% for predicting the match outcome.")
print("")
print("")  # predict prints
knn_open_pred, knn_open_test, knn_open_k, knn_open_measure = knn_ml('open_labels', full_dict, START_K, END_K, STEP_K,
                                                                    chess_df)
print("")
print(
    f"Using kNN, the k* value '{knn_open_k}' was the most accurate with an accuracy of {knn_open_measure}% for predicting chess openers.")
print("")
print("")  # predict prints
knn_whiteone_pred, knn_whiteone_test, knn_whiteone_k, knn_whiteone_measure = knn_ml('whiteopen_labels', whitethird_dict,
                                                                                    START_K, END_K, STEP_K, chess_df)
print("")
print(
    f"Using kNN, the k* value '{knn_whiteone_k}' was the most accurate with an accuracy of {knn_whiteone_measure}% for predicting White's first move.")
print("")
print("")  # predict prints
knn_blackone_pred, knn_blackone_test, knn_blackone_k, knn_blackone_measure = knn_ml('blackopen_labels', blackthird_dict,
                                                                                    START_K, END_K, STEP_K, chess_df)
print("")
print(
    f"Using kNN, the k* value '{knn_blackone_k}' was the most accurate with an accuracy of {knn_blackone_measure}% for predicting Black's response to White's first move.")
print("")
print("")

print("")
print("Analyzing use of the Indian Defense-")
print("")  # predict prints
knn_indchk_pred, knn_indchk_test, knn_indchk_k, knn_indchk_measure = knn_ml('open_labels', third_dict, START_K, END_K,
                                                                            STEP_K, indianchk_df)
print("")
print(
    f"Using kNN, the k* value '{knn_indchk_k}' was the most accurate with an accuracy of {knn_indchk_measure}% for predicting Black's usage of the Indian Defense opening.")
print("")
print("")  # predict prints
knn_indchkoutcome_pred, knn_indchkoutcome_test, knn_indchkoutcome_k, knn_indchkoutcome_measure = knn_ml('winner_labels',
                                                                                                        full_dict,
                                                                                                        START_K, END_K,
                                                                                                        STEP_K,
                                                                                                        indianchk_df)
print("")
print(
    f"Using kNN, the k* value '{knn_indchkoutcome_k}' was the most accurate with an accuracy of {knn_indchkoutcome_measure}% for predicting the match outcome looking at use of Indian Defense.")
print("")
print("")

print("")
print("Analyzing Black's resposne to 'W1: e4'-")
print("")  # predict prints
knn_e4open_pred, knn_e4open_test, knn_e4open_k, knn_e4open_measure = knn_ml('open_labels', third_dict, START_K, END_K,
                                                                            STEP_K, e4open_df)
print("")
print(
    f"Using kNN, the k* value '{knn_e4open_k}' was the most accurate with an accuracy of {knn_e4open_measure}% for predicting Black's response to 'W1: e4'.")
print("")
print("")  # predict prints
knn_e4openoutcome_pred, knn_e4openoutcome_test, knn_e4openoutcome_k, knn_e4openoutcome_measure = knn_ml('winner_labels',
                                                                                                        full_dict,
                                                                                                        START_K, END_K,
                                                                                                        STEP_K,
                                                                                                        e4open_df)
print("")
print(
    f"Using kNN, the k* value '{knn_e4openoutcome_k}' was the most accurate with an accuracy of {knn_e4openoutcome_measure}% for predicting the match outcome looking at Black's response to 'W1: e4'.")
print("")
print("")

print("")
print("Analyzing Ruy Lopez and the Italian Game-")
print("")  # predict prints
knn_ruyvsita_pred, knn_ruyvsita_test, knn_ruyvsita_k, knn_ruyvsita_measure = knn_ml('open_labels', third_dict, START_K,
                                                                                    END_K, STEP_K, ruyvsitaly_df)
print("")
print(
    f"Using kNN, the k* value '{knn_ruyvsita_k}' was the most accurate with an accuracy of {knn_ruyvsita_measure}% for predicting White's usage of Ruy Lopez, Italian Game, or other.")
print("")
print("")  # predict prints
knn_ruyvsitaoutcome_pred, knn_ruyvsitaoutcome_test, knn_ruyvsitaoutcome_k, knn_ruyvsitaoutcome_measure = knn_ml(
    'winner_labels', full_dict, START_K, END_K, STEP_K, ruyvsitaly_df)
print("")
print(
    f"Using kNN, the k* value '{knn_ruyvsitaoutcome_k}' was the most accurate with an accuracy of {knn_ruyvsitaoutcome_measure}% for predicting the match outcome of Ruy Lopez, Italian Game, and other.")
print("")
print("")

print("")
print("Analyzing the effectiveness of the Queen's Gambit-")
print("")  # predict prints
knn_queensg_pred, knn_queensg_test, knn_queensg_k, knn_queensg_measure = knn_ml('open_labels', third_dict, START_K,
                                                                                END_K, STEP_K, queens_df)
print("")
print(
    f"Using kNN, the k* value '{knn_queensg_k}' was the most accurate with an accuracy of {knn_queensg_measure}% for predicting White's usage of the Queen's Gambit.")
print("")
print("")  # predict prints
knn_queensgoutcome_pred, knn_queensgoutcome_test, knn_queensgoutcome_k, knn_queensgoutcome_measure = knn_ml(
    'winner_labels', full_dict, START_K, END_K, STEP_K, queens_df)
print("")
print(
    f"Using kNN, the k* value '{knn_queensgoutcome_k}' was the most accurate with an accuracy of {knn_queensgoutcome_measure}% for predicting the match outcome of Queen's Gambit and other.")
print("")
print("")

# Naive Bayesian
print("")
print("")
print("Naive Bayesian Analysis-")
print("")
print("")  # nbt
print("")
print("Predicting Human vs Computer, Match Outcome, Chess Openers, White First Move, and Black First Move-")
print("")  # predict prints
nbt_humanvscomp_pred, nbt_humanvscomp_test, nbt_humanvscomp_measure = nbt_ml('human_labels', full_dict, chess_df)
print("")
print(f"Using the Naive Bayesian method, the accuracy for predicting human players is {nbt_humanvscomp_measure}%.")
print("")
print("")  # predict prints
nbt_fulloutcome_pred, nbt_fulloutcome_test, nbt_fulloutcome_measure = nbt_ml('winner_labels', full_dict, chess_df)
print("")
print(f"Using the Naive Bayesian method, the accuracy of predicting the match outcome is {nbt_fulloutcome_measure}%.")
print("")
print("")  # predict prints
nbt_open_pred, nbt_open_test, nbt_open_measure = nbt_ml('open_labels', full_dict, chess_df)
print("")
print(f"Using the Naive Bayesian method, the accuracy of predicting chess openers is {nbt_open_measure}%.")
print("")
print("")  # predict prints
nbt_whiteone_pred, nbt_whiteone_test, nbt_whiteone_measure = nbt_ml('whiteopen_labels', whitethird_dict, chess_df)
print("")
print(f"Using the Naive Bayesian method, the accuracy of predicting White's first move is {nbt_whiteone_measure}%.")
print("")
print("")  # predict prints
nbt_blackone_pred, nbt_blackone_test, nbt_blackone_measure = nbt_ml('blackopen_labels', blackthird_dict, chess_df)
print("")
print(
    f"Using the Naive Bayesian method, the accuracy of predicting Black's response to White's first move is {nbt_blackone_measure}%.")
print("")
print("")

print("")
print("Analyzing use of the Indian Defense-")
print("")  # predict prints
nbt_indchk_pred, nbt_indchk_test, nbt_indchk_measure = nbt_ml('open_labels', third_dict, indianchk_df)
print("")
print(
    f"Using the Naive Bayesian method, the accuracy of predicting Black's usage of the Indian Defense opening is {nbt_indchk_measure}%.")
print("")
print("")  # predict prints
nbt_indchkoutcome_pred, nbt_indchkoutcome_test, nbt_indchkoutcome_measure = nbt_ml('winner_labels', full_dict,
                                                                                   indianchk_df)
print("")
print(
    f"Using the Naive Bayesian method, the accuracy of predicting the match outcome looking at use of Indian Defense is {nbt_indchkoutcome_measure}%.")
print("")
print("")

print("")
print("Analyzing Ruy Lopez and the Italian Game-")
print("")  # predict prints
nbt_ruyvsita_pred, nbt_ruyvsita_test, nbt_ruyvsita_measure = nbt_ml('open_labels', third_dict, ruyvsitaly_df)
print("")
print(
    f"Using the Naive Bayesian method, the accuracy of predicting White's usage of Ruy Lopez, Italian Game, or other is {nbt_ruyvsita_measure}%.")
print("")
print("")  # predict prints
nbt_ruyvsitaoutcome_pred, nbt_ruyvsitaoutcome_test, nbt_ruyvsitaoutcome_measure = nbt_ml('winner_labels', full_dict,
                                                                                         ruyvsitaly_df)
print("")
print(
    f"Using the Naive Bayesian method, the accuracy of predicting the match outcome of Ruy Lopez, Italian Game, and other is {nbt_ruyvsitaoutcome_measure}%.")
print("")
print("")

print("")
print("Analyzing Black's resposne to 'W1: e4'-")
print("")  # predict prints
nbt_e4open_pred, nbt_e4open_test, nbt_e4open_measure = nbt_ml('open_labels', third_dict, e4open_df)
print("")
print(
    f"Using the Naive Bayesian method, the accuracy of predicting Black's response to 'W1: e4' is {nbt_e4open_measure}%.")
print("")
print("")  # predict prints
nbt_e4openoutcome_pred, nbt_e4openoutcome_test, nbt_e4openoutcome_measure = nbt_ml('winner_labels', full_dict,
                                                                                   e4open_df)
print("")
print(
    f"Using the Naive Bayesian method, the accuracy of predicting the match outcome of Black's response to 'W1: e4' is {nbt_e4openoutcome_measure}%.")
print("")
print("")

print("")
print("Analyzing the effectiveness of the Queen's Gambit-")
print("")  # predict prints
nbt_queensg_pred, nbt_queensg_test, nbt_queensg_measure = nbt_ml('open_labels', third_dict, queens_df)
print("")
print(
    f"Using the Naive Bayesian method, the accuracy of predicting White's usage of the Queen's Gambit is {nbt_queensg_measure}%.")
print("")
print("")  # predict prints
nbt_queensgoutcome_pred, nbt_queensgoutcome_test, nbt_queensgoutcome_measure = nbt_ml('winner_labels', full_dict,
                                                                                      queens_df)
print("")
print(
    f"Using the Naive Bayesian method, the accuracy of predicting the match outcome of Queen's Gambit and other is {nbt_queensgoutcome_measure}%.")
print("")
print("")

# Decision Tree
print("")
print("")
print("Decision Tree Analysis-")
print("")
print("")  # dt
print("")
print("Predicting Human vs Computer, Match Outcome, Chess Openers, White First Move, and Black First Move-")
print("")  # predict prints
dt_humanvscomp_pred, dt_humanvscomp_test, dt_humanvscomp_measure = dt_ml('human_labels', full_dict, chess_df)
print("")
print(f"Using the Decision Tree method, the accuracy for predicting human players is {dt_humanvscomp_measure}%.")
print("")
print("")  # predict prints
dt_fulloutcome_pred, dt_fulloutcome_test, dt_fulloutcome_measure = dt_ml('winner_labels', full_dict, chess_df)
print("")
print(f"Using the Decision Tree method, the accuracy of predicting the match outcome is {dt_fulloutcome_measure}%.")
print("")
print("")  # predict prints
dt_open_pred, dt_open_test, dt_open_measure = dt_ml('open_labels', full_dict, chess_df)
print("")
print(f"Using the Decision Tree method, the accuracy of predicting chess openers is {dt_open_measure}%.")
print("")
print("")  # predict prints
dt_whiteone_pred, dt_whiteone_test, dt_whiteone_measure = dt_ml('whiteopen_labels', whitethird_dict, chess_df)
print("")
print(f"Using the Decision Tree method, the accuracy of predicting White's first move is {dt_whiteone_measure}%.")
print("")
print("")  # predict prints
dt_blackone_pred, dt_blackone_test, dt_blackone_measure = dt_ml('blackopen_labels', blackthird_dict, chess_df)
print("")
print(
    f"Using the Decision Tree method, the accuracy of predicting Black's response to White's first move is {dt_blackone_measure}%.")
print("")
print("")

print("")
print("Analyzing use of the Indian Defense-")
print("")  # predict prints
dt_indchk_pred, dt_indchk_test, dt_indchk_measure = dt_ml('open_labels', third_dict, indianchk_df)
print("")
print(
    f"Using the Decision Tree method, the accuracy of predicting Black's usage of the Indian Defense opening is {dt_indchk_measure}%.")
print("")
print("")  # predict prints
dt_indchkoutcome_pred, dt_indchkoutcome_test, dt_indchkoutcome_measure = dt_ml('winner_labels', full_dict, indianchk_df)
print("")
print(
    f"Using the Decision Tree method, the accuracy of predicting the match outcome looking at use of Indian Defense is {dt_indchkoutcome_measure}%.")
print("")
print("")

print("")
print("Analyzing Black's resposne to 'W1: e4'-")
print("")  # predict prints
dt_e4open_pred, dt_e4open_test, dt_e4open_measure = dt_ml('open_labels', third_dict, e4open_df)
print("")
print(
    f"Using the Decision Tree method, the accuracy of predicting Black's response to 'W1: e4' is {dt_e4open_measure}%.")
print("")
print("")  # predict prints
dt_e4openoutcome_pred, dt_e4openoutcome_test, dt_e4openoutcome_measure = dt_ml('winner_labels', full_dict, e4open_df)
print("")
print(
    f"Using the Decision Tree method, the accuracy of predicting the match outcome of Black's response to 'W1: e4' is {dt_e4openoutcome_measure}%.")
print("")
print("")

print("")
print("Analyzing Ruy Lopez and the Italian Game-")
print("")  # predict prints
dt_ruyvsita_pred, dt_ruyvsita_test, dt_ruyvsita_measure = dt_ml('open_labels', third_dict, ruyvsitaly_df)
print("")
print(
    f"Using the Decision Tree method, the accuracy of predicting White's usage of Ruy Lopez, Italian Game, or other is {dt_ruyvsita_measure}%.")
print("")
print("")  # predict prints
dt_ruyvsitaoutcome_pred, dt_ruyvsitaoutcome_test, dt_ruyvsitaoutcome_measure = dt_ml('winner_labels', full_dict,
                                                                                     ruyvsitaly_df)
print("")
print(
    f"Using the Decision Tree method, the accuracy of predicting the match outcome of Ruy Lopez, Italian Game, and other is {dt_ruyvsitaoutcome_measure}%.")
print("")
print("")

print("")
print("Analyzing the effectiveness of the Queen's Gambit-")
print("")  # predict prints
dt_queensg_pred, dt_queensg_test, dt_queensg_measure = dt_ml('open_labels', third_dict, queens_df)
print("")
print(
    f"Using the Decision Tree method, the accuracy of predicting White's usage of the Queen's Gambit is {dt_queensg_measure}%.")
print("")
print("")  # predict prints
dt_queensgoutcome_pred, dt_queensgoutcome_test, dt_queensgoutcome_measure = dt_ml('winner_labels', full_dict, queens_df)
print("")
print(
    f"Using the Decision Tree method, the accuracy of predicting the match outcome of Queen's Gambit and other is {dt_queensgoutcome_measure}%.")
print("")
print("")

# Random Forest
print("")
print("")
print("Random Forest Analysis-")
print("")
print("")  # rf
print("")
print("Predicting Human vs Computer, Match Outcome, Chess Openers, White First Move, and Black First Move-")
print("")  # predict prints
rf_humanvscomp_pred, rf_humanvscomp_test, rf_humanvscomp_optnd, rf_humanvscomp_measure = rf_ml('human_labels',
                                                                                               full_dict, chess_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_humanvscomp_optnd}' were the most accurate with an accuracy of {rf_humanvscomp_measure}% for predicting human players.")
print("")
print("")  # predict prints
rf_fulloutcome_pred, rf_fulloutcome_test, rf_fulloutcome_optnd, rf_fulloutcome_measure = rf_ml('winner_labels',
                                                                                               full_dict, chess_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_fulloutcome_optnd}' were the most accurate with an accuracy of {rf_fulloutcome_measure}% for predicting the match outcome.")
print("")
print("")  # predict prints
rf_open_pred, rf_open_test, rf_open_optnd, rf_open_measure = rf_ml('open_labels', full_dict, chess_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_open_optnd}' were the most accurate with an accuracy of {rf_open_measure}% for predicting chess openers.")
print("")
print("")  # predict prints
rf_whiteone_pred, rf_whiteone_test, rf_whiteone_optnd, rf_whiteone_measure = rf_ml('whiteopen_labels', whitethird_dict,
                                                                                   chess_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_whiteone_optnd}' were the most accurate with an accuracy of {rf_whiteone_measure}% for predicting White's first move.")
print("")
print("")  # predict prints
rf_blackone_pred, rf_blackone_test, rf_blackone_optnd, rf_blackone_measure = rf_ml('blackopen_labels', blackthird_dict,
                                                                                   chess_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_blackone_optnd}' were the most accurate with an accuracy of {rf_blackone_measure}% for predicting Black's response to White's first move.")
print("")
print("")

print("")
print("Analyzing use of the Indian Defense-")
print("")  # predict prints
rf_indchk_pred, rf_indchk_test, rf_indchk_optnd, rf_indchk_measure = rf_ml('open_labels', third_dict, indianchk_df)
print("")
print(
    f"Using Random Forest the optimal 'n' and 'd' values '{rf_indchk_optnd}' were the most accurate with an accuracy of {rf_indchk_measure}% for predicting Black's usage of the Indian Defense opening.")
print("")
print("")  # predict prints
rf_indchkoutcome_pred, rf_indchkoutcome_test, rf_indchkoutcome_optnd, rf_indchkoutcome_measure = rf_ml('winner_labels',
                                                                                                       full_dict,
                                                                                                       indianchk_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_indchkoutcome_optnd}' were the most accurate with an accuracy of {rf_indchkoutcome_measure}% for predicting the match outcome looking at use of Indian Defense.")
print("")
print("")

print("")
print("Analyzing Black's resposne to 'W1: e4'-")
print("")  # predict prints
rf_e4open_pred, rf_e4open_test, rf_e4open_optnd, rf_e4open_measure = rf_ml('open_labels', third_dict, e4open_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values'{rf_e4open_optnd}' was the most accurate with an accuracy of {rf_e4open_measure}% for predicting Black's response to 'W1: e4'.")
print("")
print("")  # predict prints
rf_e4openoutcome_pred, rf_e4openoutcome_test, rf_e4openoutcome_optnd, rf_e4openoutcome_measure = rf_ml('winner_labels',
                                                                                                       full_dict,
                                                                                                       e4open_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_e4openoutcome_optnd}' was the most accurate with an accuracy of {rf_e4openoutcome_measure}% for predicting the match outcome looking at Black's response to 'W1: e4'.")
print("")
print("")

print("")
print("Analyzing Ruy Lopez and the Italian Game-")
print("")  # predict prints
rf_ruyvsita_pred, rf_ruyvsita_test, rf_ruyvsita_optnd, rf_ruyvsita_measure = rf_ml('open_labels', third_dict,
                                                                                   ruyvsitaly_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_ruyvsita_optnd}' were the most accurate with an accuracy of {rf_ruyvsita_measure}% for predicting White's usage of Ruy Lopez, Italian Game, or other.")
print("")
print("")  # predict prints
rf_ruyvsitaoutcome_pred, rf_ruyvsitaoutcome_test, rf_ruyvsitaoutcome_optnd, rf_ruyvsitaoutcome_measure = rf_ml(
    'winner_labels', full_dict, ruyvsitaly_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_ruyvsitaoutcome_optnd}' were the most accurate with an accuracy of {rf_ruyvsitaoutcome_measure}% for predicting the match outcome of Ruy Lopez, Italian Game, and other.")
print("")
print("")

print("")
print("Analyzing the effectiveness of the Queen's Gambit-")
print("")  # predict prints
rf_queensg_pred, rf_queensg_test, rf_queensg_optnd, rf_queensg_measure = rf_ml('open_labels', third_dict, queens_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_queensg_optnd}' were the most accurate with an accuracy of {rf_queensg_measure}% for predicting White's usage of the Queen's Gambit.")
print("")
print("")  # predict prints
rf_queensgoutcome_pred, rf_queensgoutcome_test, rf_queensgoutcome_optnd, rf_queensgoutcome_measure = rf_ml(
    'winner_labels', full_dict, queens_df)
print("")
print(
    f"Using Random Forest, the optimal 'n' and 'd' values '{rf_queensgoutcome_optnd}' were the most accurate with an accuracy of {rf_queensgoutcome_measure}% for predicting the match outcome of Queen's Gambit and other.")
print("")
print("")

# Final Analysis
start_fig = plt.figure()  # analyze correct predictions of human in game and winner
ax = start_fig.add_axes([0, 0, 1, 1])
classifiers = ['kNN', 'Naive Bayesian', 'Decision Tree', 'Random Forest']  # classifiers
humanvscomplist = [knn_humanvscomp_measure, nbt_humanvscomp_measure, dt_humanvscomp_measure,
                   rf_humanvscomp_measure]  # measures
fulloutcomelist = [knn_fulloutcome_measure, nbt_fulloutcome_measure, dt_fulloutcome_measure, rf_fulloutcome_measure]
start_perc = [humanvscomplist, fulloutcomelist]
ax.set_ylabel('Percentage of Correct Classification')  # graph titles
ax.set_title('Accurate Classification Rates for Human vs Computer and Full Match Outcome')
ax.legend(labels=['Human vs Computer', 'Full Match Outcome'])
plt.xticks([p + BAR_WIDTH for p in range(len(start_perc[0]))], classifiers)  # bar chart
ax.bar(np.arange(len(classifiers)) + 0.00, start_perc[0], color='b', width=BAR_WIDTH, label='Human vs Computer')
ax.bar(np.arange(len(classifiers)) + 0.25, start_perc[1], color='r', width=BAR_WIDTH, label='Full Match Outcome')
# bar chart setup
plt.legend()
plt.show()

open_fig = plt.figure()  # analyze correct predictions for opener, white opener, and black response to white opener
ax = open_fig.add_axes([0, 0, 1, 1])
classifiers = ['kNN', 'Naive Bayesian', 'Decision Tree', 'Random Forest']  # classifiers
openlist = [knn_open_measure, nbt_open_measure, dt_open_measure, rf_open_measure]  # measures
wopenlist = [knn_whiteone_measure, nbt_whiteone_measure, dt_whiteone_measure, rf_whiteone_measure]
bopenlist = [knn_blackone_measure, nbt_blackone_measure, dt_blackone_measure, rf_blackone_measure]
open_perc = [openlist, wopenlist, bopenlist]
ax.set_ylabel('Percentage of Correct Classification')  # graph titles
ax.set_title('Accurate Classification Rates for Opener, First White Move, and First Black Move')
ax.legend(labels=['Opener', 'First White Move', 'First Black Move'])
plt.xticks([p + BAR_WIDTH for p in range(len(start_perc[0]))], classifiers)  # bar chart
ax.bar(np.arange(len(classifiers)) + 0.00, open_perc[0], color='b', width=BAR_WIDTH, label='Opener')
ax.bar(np.arange(len(classifiers)) + 0.25, open_perc[1], color='g', width=BAR_WIDTH, label='First White Move')
ax.bar(np.arange(len(classifiers)) + 0.50, open_perc[2], color='r', width=BAR_WIDTH, label='First Black Move')
# bar chart setup
plt.legend()
plt.show()

indchk_fig = plt.figure()  # analyze correct predictions for use and match outcome of Indian Defense vs Other
ax = indchk_fig.add_axes([0, 0, 1, 1])
classifiers = ['kNN', 'Naive Bayesian', 'Decision Tree', 'Random Forest']  # classifiers
indchklist = [knn_indchk_measure, nbt_indchk_measure, dt_indchk_measure, rf_indchk_measure]  # measures
indchkolist = [knn_indchkoutcome_measure, nbt_indchkoutcome_measure, dt_indchkoutcome_measure, rf_indchkoutcome_measure]
indchk_perc = [indchklist, indchkolist]
ax.set_ylabel('Percentage of Correct Classification')  # graph titles
ax.set_title('Accurate Classification Rates for Indian Defense Usage and Match Outcome')
ax.legend(labels=['Indian Defense Usage', 'Indian Defense Outcome'])
plt.xticks([p + BAR_WIDTH for p in range(len(indchk_perc[0]))], classifiers)  # bar chart
ax.bar(np.arange(len(classifiers)) + 0.00, indchk_perc[0], color='b', width=BAR_WIDTH, label='Indian Defense Usage')
ax.bar(np.arange(len(classifiers)) + 0.25, indchk_perc[1], color='r', width=BAR_WIDTH, label='Indian Defense Outcome')
# bar chart setup
plt.legend()
plt.show()

ruyvsita_fig = plt.figure()  # analyze correct predictions for use and match outcome of Ruy Lopez vs Indian Game vs Other
ax = ruyvsita_fig.add_axes([0, 0, 1, 1])
classifiers = ['kNN', 'Naive Bayesian', 'Decision Tree', 'Random Forest']  # classifiers
ruyvsitalist = [knn_ruyvsita_measure, nbt_ruyvsita_measure, dt_ruyvsita_measure, rf_ruyvsita_measure]  # measure
ruyvsitaolist = [knn_ruyvsitaoutcome_measure, nbt_ruyvsitaoutcome_measure, dt_ruyvsitaoutcome_measure,
                 rf_ruyvsitaoutcome_measure]
ruyvsita_perc = [ruyvsitalist, ruyvsitaolist]
ax.set_ylabel('Percentage of Correct Classification')  # graph titles
ax.set_title('Accurate Classification Rates for Ruy Lopez/Italian Game Usage and Match Outcome')
ax.legend(labels=['Ruy Lopez/Italian Game Usage', 'Ruy Lopez/Italian Game Outcome'])
plt.xticks([p + BAR_WIDTH for p in range(len(ruyvsita_perc[0]))], classifiers)  # bar chart
ax.bar(np.arange(len(classifiers)) + 0.00, ruyvsita_perc[0], color='b', width=BAR_WIDTH,
       label='Ruy Lopez/Italian Game Usage')
ax.bar(np.arange(len(classifiers)) + 0.25, ruyvsita_perc[1], color='r', width=BAR_WIDTH,
       label='Ruy Lopez/Italian Game Usage Outcome')
# bar chart setup
plt.legend()
plt.show()

e4open_fig = plt.figure()  # analyze correct predictions for use and match outcome of responses to 'W1: e4'
ax = e4open_fig.add_axes([0, 0, 1, 1])
classifiers = ['kNN', 'Naive Bayesian', 'Decision Tree', 'Random Forest']  # classifiers
e4openlist = [knn_e4open_measure, nbt_e4open_measure, dt_e4open_measure, rf_e4open_measure]  # measures
e4openolist = [knn_e4openoutcome_measure, nbt_e4openoutcome_measure, dt_e4openoutcome_measure, rf_e4openoutcome_measure]
e4open_perc = [e4openlist, e4openolist]
ax.set_ylabel('Percentage of Correct Classification')  # graph titles
ax.set_title('Accurate Classification Rates for Black Response to ''w1= e4'' and Match Outcome')
ax.legend(labels=['Black Response to ''w1= e4', 'Black Response to ''w1= e4'' Outcome'])
plt.xticks([p + BAR_WIDTH for p in range(len(e4open_perc[0]))], classifiers)  # bar chart
ax.bar(np.arange(len(classifiers)) + 0.00, e4open_perc[0], color='b', width=BAR_WIDTH,
       label='Black Response to ''w1= e4'' Usage')
ax.bar(np.arange(len(classifiers)) + 0.25, e4open_perc[1], color='r', width=BAR_WIDTH,
       label='Black Response to ''w1= e4'' Outcome')
# bar chart setup
plt.legend()
plt.show()

queensg_fig = plt.figure()  # analyze correct predictions for use and match outcome of Queen's Gambit
ax = queensg_fig.add_axes([0, 0, 1, 1])
classifiers = ['kNN', 'Naive Bayesian', 'Decision Tree', 'Random Forest']  # classifiers
queensglist = [knn_queensg_measure, nbt_queensg_measure, dt_queensg_measure, rf_queensg_measure]  # measures
queensgolist = [knn_queensgoutcome_measure, nbt_queensgoutcome_measure, dt_queensgoutcome_measure,
                rf_queensgoutcome_measure]
queensg_perc = [queensglist, queensgolist]
ax.set_ylabel('Percentage of Correct Classification')  # graph titles
ax.set_title('Accurate Classification Rates for Queen''s Gambit Usage and Match Outcome')
ax.legend(labels=['Queen''s Gambit Usage', 'Queen''s Gambit Outcome'])
plt.xticks([p + BAR_WIDTH for p in range(len(queensg_perc[0]))], classifiers)  # bar chart
ax.bar(np.arange(len(classifiers)) + 0.00, queensg_perc[0], color='b', width=BAR_WIDTH, label='Queen''s Gambit Usage')
ax.bar(np.arange(len(classifiers)) + 0.25, queensg_perc[1], color='r', width=BAR_WIDTH,
       label='Queen''s Gambit Usage Outcome')
# bar chart setup
plt.legend()
plt.show()

# Classification Report
print("")
print("")
print("Classification Reports-")  # classification reports with precision, recall, and f1-score
print("")
print("")
print("Human vs Computer Classification-")
print("")
print("")
knnstring = 'kNN '  # strings for classifiers
nbtstring = 'Naive Bayesian '
dtstring = 'Decision Tree '
rfstring = 'Random Forest '
humanvscomppred_list = []  # blank list
for e in np.unique(knn_humanvscomp_pred):
    # use unique values with prediction array along with classifier string to name index rows
    humanvscomppred_list.append(knnstring + list(HUMAN_LABELS_DICT.keys())[list(HUMAN_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_humanvscomp_pred):
    humanvscomppred_list.append(nbtstring + list(HUMAN_LABELS_DICT.keys())[list(HUMAN_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_humanvscomp_pred):
    humanvscomppred_list.append(dtstring + list(HUMAN_LABELS_DICT.keys())[list(HUMAN_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_humanvscomp_pred):
    humanvscomppred_list.append(rfstring + list(HUMAN_LABELS_DICT.keys())[list(HUMAN_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
humanvscomp_knncr = ((pd.DataFrame.from_dict(classification_report(knn_humanvscomp_test,
                                                                   knn_humanvscomp_pred,
                                                                   labels=np.unique(knn_humanvscomp_pred),
                                                                   output_dict=True))).T).head(
    len(np.unique(knn_humanvscomp_pred)))
humanvscomp_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_humanvscomp_test,
                                                                   nbt_humanvscomp_pred,
                                                                   labels=np.unique(nbt_humanvscomp_pred),
                                                                   output_dict=True))).T).head(
    len(np.unique(nbt_humanvscomp_pred)))
humanvscomp_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_humanvscomp_test,
                                                                  dt_humanvscomp_pred,
                                                                  labels=np.unique(dt_humanvscomp_pred),
                                                                  output_dict=True))).T).head(
    len(np.unique(dt_humanvscomp_pred)))
humanvscomp_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_humanvscomp_test,
                                                                  rf_humanvscomp_pred,
                                                                  labels=np.unique(rf_humanvscomp_pred),
                                                                  output_dict=True))).T).head(
    len(np.unique(rf_humanvscomp_pred)))
humanvscomp_totalcr = (
            (pd.concat([humanvscomp_knncr, humanvscomp_nbtcr, humanvscomp_dtcr, humanvscomp_rfcr])) * 100).round(ROUND)
humanvscomp_totalcr = humanvscomp_totalcr.drop(columns=['support'])
humanvscomp_totalcr['Class Prediction'] = humanvscomppred_list
humanvscomp_totalcr = humanvscomp_totalcr.set_index('Class Prediction')
print(humanvscomp_totalcr)  # print
print("")
print("")
print("Full Match Outcome Classification-")
print("")
print("")
fulloutcomepred_list = []
for e in np.unique(knn_fulloutcome_pred):
    # use unique values with prediction array along with classifier string to name index rows
    fulloutcomepred_list.append(knnstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_fulloutcome_pred):
    fulloutcomepred_list.append(nbtstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_fulloutcome_pred):
    fulloutcomepred_list.append(dtstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_fulloutcome_pred):
    fulloutcomepred_list.append(rfstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
fulloutcome_knncr = ((pd.DataFrame.from_dict(classification_report(knn_fulloutcome_test,
                                                                   knn_fulloutcome_pred,
                                                                   labels=np.unique(knn_fulloutcome_pred),
                                                                   output_dict=True))).T).head(
    len(np.unique(knn_fulloutcome_pred)))
fulloutcome_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_fulloutcome_test,
                                                                   nbt_fulloutcome_pred,
                                                                   labels=np.unique(nbt_fulloutcome_pred),
                                                                   output_dict=True))).T).head(
    len(np.unique(nbt_fulloutcome_pred)))
fulloutcome_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_fulloutcome_test,
                                                                  dt_fulloutcome_pred,
                                                                  labels=np.unique(dt_fulloutcome_pred),
                                                                  output_dict=True))).T).head(
    len(np.unique(dt_fulloutcome_pred)))
fulloutcome_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_fulloutcome_test,
                                                                  rf_fulloutcome_pred,
                                                                  labels=np.unique(rf_fulloutcome_pred),
                                                                  output_dict=True))).T).head(
    len(np.unique(rf_fulloutcome_pred)))
fulloutcome_totalcr = (
            (pd.concat([fulloutcome_knncr, fulloutcome_nbtcr, fulloutcome_dtcr, fulloutcome_rfcr])) * 100).round(ROUND)
fulloutcome_totalcr = fulloutcome_totalcr.drop(
    columns=['support'])  # combine all classification reports for each classifier
fulloutcome_totalcr['Class Prediction'] = fulloutcomepred_list
fulloutcome_totalcr = fulloutcome_totalcr.set_index('Class Prediction')
print(fulloutcome_totalcr)  # print
print("")
print("")
print("Opener Classification-")
print("")
print("")
openpred_list = []
for e in np.unique(knn_open_pred):
    # use unique values with prediction array along with classifier string to name index rows
    openpred_list.append(knnstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_open_pred):
    openpred_list.append(nbtstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_open_pred):
    openpred_list.append(dtstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_open_pred):
    openpred_list.append(rfstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
open_knncr = ((pd.DataFrame.from_dict(classification_report(knn_open_test,
                                                            knn_open_pred, labels=np.unique(knn_open_pred),
                                                            output_dict=True))).T).head(len(np.unique(knn_open_pred)))
open_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_open_test,
                                                            nbt_open_pred, labels=np.unique(nbt_open_pred),
                                                            output_dict=True))).T).head(len(np.unique(nbt_open_pred)))
open_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_open_test,
                                                           dt_open_pred, labels=np.unique(dt_open_pred),
                                                           output_dict=True))).T).head(len(np.unique(dt_open_pred)))
open_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_open_test,
                                                           rf_open_pred, labels=np.unique(rf_open_pred),
                                                           output_dict=True))).T).head(len(np.unique(rf_open_pred)))
open_totalcr = ((pd.concat([open_knncr, open_nbtcr, open_dtcr, open_rfcr])) * 100).round(ROUND)
open_totalcr = open_totalcr.drop(columns=['support'])  # combine all classification reports for each classifier
open_totalcr['Class Prediction'] = openpred_list
open_totalcr = open_totalcr.set_index('Class Prediction')
print(open_totalcr)  # print
print("")
print("")
print("White's First Move Classification")
print("")
print("")
whiteonepred_list = []
for e in np.unique(knn_whiteone_pred):
    # use unique values with prediction array along with classifier string to name index rows
    whiteonepred_list.append(
        knnstring + list(WHITEOPEN_LABELS_DICT.keys())[list(WHITEOPEN_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_whiteone_pred):
    whiteonepred_list.append(
        nbtstring + list(WHITEOPEN_LABELS_DICT.keys())[list(WHITEOPEN_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_whiteone_pred):
    whiteonepred_list.append(
        dtstring + list(WHITEOPEN_LABELS_DICT.keys())[list(WHITEOPEN_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_whiteone_pred):
    whiteonepred_list.append(
        rfstring + list(WHITEOPEN_LABELS_DICT.keys())[list(WHITEOPEN_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
whiteone_knncr = ((pd.DataFrame.from_dict(classification_report(knn_whiteone_test,
                                                                knn_whiteone_pred, labels=np.unique(knn_whiteone_pred),
                                                                output_dict=True))).T).head(
    len(np.unique(knn_whiteone_pred)))
whiteone_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_whiteone_test,
                                                                nbt_whiteone_pred, labels=np.unique(nbt_whiteone_pred),
                                                                output_dict=True))).T).head(
    len(np.unique(nbt_whiteone_pred)))
whiteone_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_whiteone_test,
                                                               dt_whiteone_pred, labels=np.unique(dt_whiteone_pred),
                                                               output_dict=True))).T).head(
    len(np.unique(dt_whiteone_pred)))
whiteone_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_whiteone_test,
                                                               rf_whiteone_pred, labels=np.unique(rf_whiteone_pred),
                                                               output_dict=True))).T).head(
    len(np.unique(rf_whiteone_pred)))
whiteone_totalcr = ((pd.concat([whiteone_knncr, whiteone_nbtcr, whiteone_dtcr, whiteone_rfcr])) * 100).round(ROUND)
whiteone_totalcr = whiteone_totalcr.drop(columns=['support'])  # combine all classification reports for each classifier
whiteone_totalcr['Class Prediction'] = whiteonepred_list
whiteone_totalcr = whiteone_totalcr.set_index('Class Prediction')
print(whiteone_totalcr)  # print
print("")
print("")
print("Black's First Move Classification")
print("")
print("")
blackonepred_list = []
for e in np.unique(knn_blackone_pred):
    # use unique values with prediction array along with classifier string to name index rows
    blackonepred_list.append(
        knnstring + list(BLACKOPEN_LABELS_DICT.keys())[list(BLACKOPEN_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_blackone_pred):
    blackonepred_list.append(
        nbtstring + list(BLACKOPEN_LABELS_DICT.keys())[list(BLACKOPEN_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_blackone_pred):
    blackonepred_list.append(
        dtstring + list(BLACKOPEN_LABELS_DICT.keys())[list(BLACKOPEN_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_blackone_pred):
    blackonepred_list.append(
        rfstring + list(BLACKOPEN_LABELS_DICT.keys())[list(BLACKOPEN_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
blackone_knncr = ((pd.DataFrame.from_dict(classification_report(knn_blackone_test,
                                                                knn_blackone_pred, labels=np.unique(knn_blackone_pred),
                                                                output_dict=True))).T).head(
    len(np.unique(knn_blackone_pred)))
blackone_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_blackone_test,
                                                                nbt_blackone_pred, labels=np.unique(nbt_blackone_pred),
                                                                output_dict=True))).T).head(
    len(np.unique(nbt_blackone_pred)))
blackone_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_blackone_test,
                                                               dt_blackone_pred, labels=np.unique(dt_blackone_pred),
                                                               output_dict=True))).T).head(
    len(np.unique(dt_blackone_pred)))
blackone_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_blackone_test,
                                                               rf_blackone_pred, labels=np.unique(rf_blackone_pred),
                                                               output_dict=True))).T).head(
    len(np.unique(rf_blackone_pred)))
blackone_totalcr = ((pd.concat([blackone_knncr, blackone_nbtcr, blackone_dtcr, blackone_rfcr])) * 100).round(ROUND)
blackone_totalcr = blackone_totalcr.drop(columns=['support'])  # combine all classification reports for each classifier
blackone_totalcr['Class Prediction'] = blackonepred_list
blackone_totalcr = blackone_totalcr.set_index('Class Prediction')
print(blackone_totalcr)  # print
print("")
print("Use of Indian Defense Classification")
print("")
print("")
indchkpred_list = []
for e in np.unique(knn_indchk_pred):
    # use unique values with prediction array along with classifier string to name index rows
    indchkpred_list.append(knnstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_indchk_pred):
    indchkpred_list.append(nbtstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_indchk_pred):
    indchkpred_list.append(dtstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_indchk_pred):
    indchkpred_list.append(rfstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
indchk_knncr = ((pd.DataFrame.from_dict(classification_report(knn_indchk_test,
                                                              knn_indchk_pred, labels=np.unique(knn_indchk_pred),
                                                              output_dict=True))).T).head(
    len(np.unique(knn_indchk_pred)))
indchk_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_indchk_test,
                                                              nbt_indchk_pred, labels=np.unique(nbt_indchk_pred),
                                                              output_dict=True))).T).head(
    len(np.unique(nbt_indchk_pred)))
indchk_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_indchk_test,
                                                             dt_indchk_pred, labels=np.unique(dt_indchk_pred),
                                                             output_dict=True))).T).head(len(np.unique(dt_indchk_pred)))
indchk_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_indchk_test,
                                                             rf_indchk_pred, labels=np.unique(rf_indchk_pred),
                                                             output_dict=True))).T).head(len(np.unique(rf_indchk_pred)))
indchk_totalcr = ((pd.concat([indchk_knncr, indchk_nbtcr, indchk_dtcr, indchk_rfcr])) * 100).round(ROUND)
indchk_totalcr = indchk_totalcr.drop(columns=['support'])  # combine all classification reports for each classifier
indchk_totalcr['Class Prediction'] = indchkpred_list
indchk_totalcr = indchk_totalcr.set_index('Class Prediction')
print(indchk_totalcr)  # print
print("")
print("")
print("Match Outcome of Indian Defense Classification")
print("")
print("")
indchkoutcomepred_list = []
for e in np.unique(knn_indchkoutcome_pred):
    # use unique values with prediction array along with classifier string to name index rows
    indchkoutcomepred_list.append(
        knnstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_indchkoutcome_pred):
    indchkoutcomepred_list.append(
        nbtstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_indchkoutcome_pred):
    indchkoutcomepred_list.append(
        dtstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_indchkoutcome_pred):
    indchkoutcomepred_list.append(
        rfstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
indchkoutcome_knncr = ((pd.DataFrame.from_dict(classification_report(knn_indchkoutcome_test,
                                                                     knn_indchkoutcome_pred,
                                                                     labels=np.unique(knn_indchkoutcome_pred),
                                                                     output_dict=True))).T).head(
    len(np.unique(knn_indchkoutcome_pred)))
indchkoutcome_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_indchkoutcome_test,
                                                                     nbt_indchkoutcome_pred,
                                                                     labels=np.unique(nbt_indchkoutcome_pred),
                                                                     output_dict=True))).T).head(
    len(np.unique(nbt_indchkoutcome_pred)))
indchkoutcome_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_indchkoutcome_test,
                                                                    dt_indchkoutcome_pred,
                                                                    labels=np.unique(dt_indchkoutcome_pred),
                                                                    output_dict=True))).T).head(
    len(np.unique(dt_indchkoutcome_pred)))
indchkoutcome_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_indchkoutcome_test,
                                                                    rf_indchkoutcome_pred,
                                                                    labels=np.unique(rf_indchkoutcome_pred),
                                                                    output_dict=True))).T).head(
    len(np.unique(rf_indchkoutcome_pred)))
indchkoutcome_totalcr = ((pd.concat(
    [indchkoutcome_knncr, indchkoutcome_nbtcr, indchkoutcome_dtcr, indchkoutcome_rfcr])) * 100).round(ROUND)
indchkoutcome_totalcr = indchkoutcome_totalcr.drop(
    columns=['support'])  # combine all classification reports for each classifier
indchkoutcome_totalcr['Class Prediction'] = indchkoutcomepred_list
indchkoutcome_totalcr = indchkoutcome_totalcr.set_index('Class Prediction')
print(indchkoutcome_totalcr)  # print
print("")
print("")
print("Use of Ruy Lopez & Italian Game Classification")
print("")
print("")
ruyvsitapred_list = []
for e in np.unique(knn_ruyvsita_pred):
    # use unique values with prediction array along with classifier string to name index rows
    ruyvsitapred_list.append(knnstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_ruyvsita_pred):
    ruyvsitapred_list.append(nbtstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_ruyvsita_pred):
    ruyvsitapred_list.append(dtstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_ruyvsita_pred):
    ruyvsitapred_list.append(rfstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
ruyvsita_knncr = ((pd.DataFrame.from_dict(classification_report(knn_ruyvsita_test,
                                                                knn_ruyvsita_pred, labels=np.unique(knn_ruyvsita_pred),
                                                                output_dict=True))).T).head(
    len(np.unique(knn_ruyvsita_pred)))
ruyvsita_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_ruyvsita_test,
                                                                nbt_ruyvsita_pred, labels=np.unique(nbt_ruyvsita_pred),
                                                                output_dict=True))).T).head(
    len(np.unique(nbt_ruyvsita_pred)))
ruyvsita_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_ruyvsita_test,
                                                               dt_ruyvsita_pred, labels=np.unique(dt_ruyvsita_pred),
                                                               output_dict=True))).T).head(
    len(np.unique(dt_ruyvsita_pred)))
ruyvsita_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_ruyvsita_test,
                                                               rf_ruyvsita_pred, labels=np.unique(rf_ruyvsita_pred),
                                                               output_dict=True))).T).head(
    len(np.unique(rf_ruyvsita_pred)))
ruyvsita_totalcr = ((pd.concat([ruyvsita_knncr, ruyvsita_nbtcr, ruyvsita_dtcr, ruyvsita_rfcr])) * 100).round(ROUND)
ruyvsita_totalcr = ruyvsita_totalcr.drop(columns=['support'])  # combine all classification reports for each classifier
ruyvsita_totalcr['Class Prediction'] = ruyvsitapred_list
ruyvsita_totalcr = ruyvsita_totalcr.set_index('Class Prediction')
print(ruyvsita_totalcr)  # print
print("")
print("")
print("Match Outcome of Ruy Lopez & Italian Game Classification")
print("")
print("")
ruyvsitaoutcomepred_list = []
for e in np.unique(knn_ruyvsitaoutcome_pred):
    # use unique values with prediction array along with classifier string to name index rows
    ruyvsitaoutcomepred_list.append(
        knnstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_ruyvsitaoutcome_pred):
    ruyvsitaoutcomepred_list.append(
        nbtstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_ruyvsitaoutcome_pred):
    ruyvsitaoutcomepred_list.append(
        dtstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_ruyvsitaoutcome_pred):
    ruyvsitaoutcomepred_list.append(
        rfstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
ruyvsitaoutcome_knncr = ((pd.DataFrame.from_dict(classification_report(knn_ruyvsitaoutcome_test,
                                                                       knn_ruyvsitaoutcome_pred,
                                                                       labels=np.unique(knn_ruyvsitaoutcome_pred),
                                                                       output_dict=True))).T).head(
    len(np.unique(knn_ruyvsitaoutcome_pred)))
ruyvsitaoutcome_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_ruyvsitaoutcome_test,
                                                                       nbt_ruyvsitaoutcome_pred,
                                                                       labels=np.unique(nbt_ruyvsitaoutcome_pred),
                                                                       output_dict=True))).T).head(
    len(np.unique(nbt_ruyvsitaoutcome_pred)))
ruyvsitaoutcome_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_ruyvsitaoutcome_test,
                                                                      dt_ruyvsitaoutcome_pred,
                                                                      labels=np.unique(dt_ruyvsitaoutcome_pred),
                                                                      output_dict=True))).T).head(
    len(np.unique(dt_ruyvsitaoutcome_pred)))
ruyvsitaoutcome_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_ruyvsitaoutcome_test,
                                                                      rf_ruyvsitaoutcome_pred,
                                                                      labels=np.unique(rf_ruyvsitaoutcome_pred),
                                                                      output_dict=True))).T).head(
    len(np.unique(rf_ruyvsitaoutcome_pred)))
ruyvsitaoutcome_totalcr = ((pd.concat(
    [ruyvsitaoutcome_knncr, ruyvsitaoutcome_nbtcr, ruyvsitaoutcome_dtcr, ruyvsitaoutcome_rfcr])) * 100).round(ROUND)
ruyvsitaoutcome_totalcr = ruyvsitaoutcome_totalcr.drop(
    columns=['support'])  # combine all classification reports for each classifier
ruyvsitaoutcome_totalcr['Class Prediction'] = ruyvsitaoutcomepred_list
ruyvsitaoutcome_totalcr = ruyvsitaoutcome_totalcr.set_index('Class Prediction')
print(ruyvsitaoutcome_totalcr)  # print
print("")
print("")
print("Use of Black Response to 'W1: e4' Classification")
print("")
print("")
e4openpred_list = []
for e in np.unique(knn_e4open_pred):
    # use unique values with prediction array along with classifier string to name index rows
    e4openpred_list.append(knnstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_e4open_pred):
    e4openpred_list.append(nbtstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_e4open_pred):
    e4openpred_list.append(dtstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_e4open_pred):
    e4openpred_list.append(rfstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
e4open_knncr = ((pd.DataFrame.from_dict(classification_report(knn_e4open_test,
                                                              knn_e4open_pred, labels=np.unique(knn_e4open_pred),
                                                              output_dict=True))).T).head(
    len(np.unique(knn_e4open_pred)))
e4open_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_e4open_test,
                                                              nbt_e4open_pred, labels=np.unique(nbt_e4open_pred),
                                                              output_dict=True))).T).head(
    len(np.unique(nbt_e4open_pred)))
e4open_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_e4open_test,
                                                             dt_e4open_pred, labels=np.unique(dt_e4open_pred),
                                                             output_dict=True))).T).head(len(np.unique(dt_e4open_pred)))
e4open_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_e4open_test,
                                                             rf_e4open_pred, labels=np.unique(rf_e4open_pred),
                                                             output_dict=True))).T).head(len(np.unique(rf_e4open_pred)))
e4open_totalcr = ((pd.concat([e4open_knncr, e4open_nbtcr, e4open_dtcr, e4open_rfcr])) * 100).round(ROUND)
e4open_totalcr = e4open_totalcr.drop(columns=['support'])  # combine all classification reports for each classifier
e4open_totalcr['Class Prediction'] = e4openpred_list
e4open_totalcr = e4open_totalcr.set_index('Class Prediction')
print(e4open_totalcr)  # print
print("")
print("")
print("Match Outcome of Black's Response to 'W1: e4' Classification")
print("")
print("")
e4openoutcomepred_list = []
for e in np.unique(knn_e4openoutcome_pred):
    # use unique values with prediction array along with classifier string to name index rows
    e4openoutcomepred_list.append(
        knnstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_e4openoutcome_pred):
    e4openoutcomepred_list.append(
        nbtstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_e4openoutcome_pred):
    e4openoutcomepred_list.append(
        dtstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_e4openoutcome_pred):
    e4openoutcomepred_list.append(
        rfstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
e4openoutcome_knncr = ((pd.DataFrame.from_dict(classification_report(knn_e4openoutcome_test,
                                                                     knn_e4openoutcome_pred,
                                                                     labels=np.unique(knn_e4openoutcome_pred),
                                                                     output_dict=True))).T).head(
    len(np.unique(knn_e4openoutcome_pred)))
e4openoutcome_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_e4openoutcome_test,
                                                                     nbt_e4openoutcome_pred,
                                                                     labels=np.unique(nbt_e4openoutcome_pred),
                                                                     output_dict=True))).T).head(
    len(np.unique(nbt_e4openoutcome_pred)))
e4openoutcome_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_e4openoutcome_test,
                                                                    dt_e4openoutcome_pred,
                                                                    labels=np.unique(dt_e4openoutcome_pred),
                                                                    output_dict=True))).T).head(
    len(np.unique(dt_e4openoutcome_pred)))
e4openoutcome_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_e4openoutcome_test,
                                                                    rf_e4openoutcome_pred,
                                                                    labels=np.unique(rf_e4openoutcome_pred),
                                                                    output_dict=True))).T).head(
    len(np.unique(rf_e4openoutcome_pred)))
e4openoutcome_totalcr = ((pd.concat(
    [e4openoutcome_knncr, e4openoutcome_nbtcr, e4openoutcome_dtcr, e4openoutcome_rfcr])) * 100).round(ROUND)
e4openoutcome_totalcr = e4openoutcome_totalcr.drop(
    columns=['support'])  # combine all classification reports for each classifier
e4openoutcome_totalcr['Class Prediction'] = e4openoutcomepred_list
e4openoutcome_totalcr = e4openoutcome_totalcr.set_index('Class Prediction')
print(e4openoutcome_totalcr)  # print
print("")
print("")
print("Use of Queen's Gambit Classification")
print("")
print("")
queensgpred_list = []
for e in np.unique(knn_queensg_pred):
    # use unique values with prediction array along with classifier string to name index rows
    queensgpred_list.append(knnstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_queensg_pred):
    queensgpred_list.append(nbtstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_queensg_pred):
    queensgpred_list.append(dtstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_queensg_pred):
    queensgpred_list.append(rfstring + list(OPENING_LABELS_DICT.keys())[list(OPENING_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
queensg_knncr = ((pd.DataFrame.from_dict(classification_report(knn_queensg_test,
                                                               knn_queensg_pred, labels=np.unique(knn_queensg_pred),
                                                               output_dict=True))).T).head(
    len(np.unique(knn_queensg_pred)))
queensg_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_queensg_test,
                                                               nbt_queensg_pred, labels=np.unique(nbt_queensg_pred),
                                                               output_dict=True))).T).head(
    len(np.unique(nbt_queensg_pred)))
queensg_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_queensg_test,
                                                              dt_queensg_pred, labels=np.unique(dt_queensg_pred),
                                                              output_dict=True))).T).head(
    len(np.unique(dt_queensg_pred)))
queensg_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_queensg_test,
                                                              rf_queensg_pred, labels=np.unique(rf_queensg_pred),
                                                              output_dict=True))).T).head(
    len(np.unique(rf_queensg_pred)))
queensg_totalcr = ((pd.concat([queensg_knncr, queensg_nbtcr, queensg_dtcr, queensg_rfcr])) * 100).round(ROUND)
queensg_totalcr = queensg_totalcr.drop(columns=['support'])  # combine all classification reports for each classifier
queensg_totalcr['Class Prediction'] = queensgpred_list
queensg_totalcr = queensg_totalcr.set_index('Class Prediction')
print(queensg_totalcr)  # print
print("")
print("")
print("Match Outcome of Queen's Gambit Classification")
print("")
print("")
queensgoutcomepred_list = []
for e in np.unique(knn_queensgoutcome_pred):
    # use unique values with prediction array along with classifier string to name index rows
    queensgoutcomepred_list.append(
        knnstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(nbt_queensgoutcome_pred):
    queensgoutcomepred_list.append(
        nbtstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(dt_queensgoutcome_pred):
    queensgoutcomepred_list.append(
        dtstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
for e in np.unique(rf_queensgoutcome_pred):
    queensgoutcomepred_list.append(
        rfstring + list(WINNER_LABELS_DICT.keys())[list(WINNER_LABELS_DICT.values()).index(e)])
    # create individual cr per classifier
queensgoutcome_knncr = ((pd.DataFrame.from_dict(classification_report(knn_queensgoutcome_test,
                                                                      knn_queensgoutcome_pred,
                                                                      labels=np.unique(knn_queensgoutcome_pred),
                                                                      output_dict=True))).T).head(
    len(np.unique(knn_queensgoutcome_pred)))
queensgoutcome_nbtcr = ((pd.DataFrame.from_dict(classification_report(nbt_queensgoutcome_test,
                                                                      nbt_queensgoutcome_pred,
                                                                      labels=np.unique(nbt_queensgoutcome_pred),
                                                                      output_dict=True))).T).head(
    len(np.unique(nbt_queensgoutcome_pred)))
queensgoutcome_dtcr = ((pd.DataFrame.from_dict(classification_report(dt_queensgoutcome_test,
                                                                     dt_queensgoutcome_pred,
                                                                     labels=np.unique(dt_queensgoutcome_pred),
                                                                     output_dict=True))).T).head(
    len(np.unique(dt_queensgoutcome_pred)))
queensgoutcome_rfcr = ((pd.DataFrame.from_dict(classification_report(rf_queensgoutcome_test,
                                                                     rf_queensgoutcome_pred,
                                                                     labels=np.unique(rf_queensgoutcome_pred),
                                                                     output_dict=True))).T).head(
    len(np.unique(rf_queensgoutcome_pred)))
queensgoutcome_totalcr = ((pd.concat(
    [queensgoutcome_knncr, queensgoutcome_nbtcr, queensgoutcome_dtcr, queensgoutcome_rfcr])) * 100).round(ROUND)
queensgoutcome_totalcr = queensgoutcome_totalcr.drop(
    columns=['support'])  # combine all classification reports for each classifier
queensgoutcome_totalcr['Class Prediction'] = queensgoutcomepred_list
queensgoutcome_totalcr = queensgoutcome_totalcr.set_index('Class Prediction')
print(queensgoutcome_totalcr)  # print
