DOMAIN2ABBR = {
    'cs': 'Computer Science',
    'econ': 'Economics',
    'eess': 'Electrical Engineering and Systems Science',
    'math': 'Mathematics',
    'physics': 'Physics',
    'q-bio': 'Quantitative Biology',
    'q-fin': 'Quantitative Finance',
    'stat': 'Statistics'
}

NUM2YEAR = {
    '20': '2020',
    '21': '2021',
    '22': '2022',
    '23': '2023'
}

def QNUM2QTYPE(qnum):
    if qnum in [1,2,3,4,5,6,7]:
        return 'Information Extraction'
    elif qnum in [8,9,13,14,15]:
        return 'Enumeration'
    elif qnum in [11,16,18]:
        return 'Pattern Recognition'
    elif qnum in [10,12,19]:
        return 'Counting'
    elif qnum in [17]:
        return 'Compositionality'
    else:
        raise ValueError(f"Invalid qnum: {qnum}")

def NUMSUBPLOTS2SUBPLOTTYPE(num_subplots):
    if num_subplots == 1:
        return '1 Subplot'
    elif 2 <= num_subplots <= 4:
        return '2-4 Subplots'
    elif num_subplots >= 5:
        return '5+ Subplots'
    else:
        raise ValueError(f"Invalid num_subplots: {num_subplots}")

IDX2ANSTYPE = {
    1: 'Text-in-Chart',
    2: 'Text-in-General',
    3: 'Number-in-Chart',
    4: 'Number-in-General'
}

IDX2SRC = {
    1: 'GPT-Sourced',
    2: 'GPT-Inspired',
    3: 'Completely Human'
}

def D_TEMPLATE():
    return {
        'Overall Score': [],

        'By Question': {
            'Q1': [],
            'Q2': [],
            'Q3': [],
            'Q4': [],
            'Q5': [],
            'Q6': [],
            'Q7': [],
            'Q8': [],
            'Q9': [],
            'Q10': [],
            'Q11': [],
            'Q12': [],
            'Q13': [],
            'Q14': [],
            'Q15': [],
            'Q16': [],
            'Q17': [],
            'Q18': [],
            'Q19': [],
        },

        'By Category': {
            'Information Extraction': [],
            'Enumeration': [],
            'Pattern Recognition': [],
            'Counting': [],
            'Compositionality': [],
        },

        'By Subplot': {
            '1 Subplot': [], 
            '2-4 Subplots': [], 
            '5+ Subplots': [], 
        },

        'By Subject': {
            'Computer Science': [],
            'Economics': [],
            'Electrical Engineering and Systems Science': [],
            'Mathematics': [],
            'Physics': [],
            'Quantitative Biology': [],
            'Quantitative Finance': [],
            'Statistics': [],
        },

        'By Year': {
            '2020': [],
            '2021': [],
            '2022': [],
            '2023': [],
        },
        
        'N_valid': [],
        'N_invalid': []
    }

def R_TEMPLATE():
    return {
    'Overall Score': [],

    'By Answer Type': {
        'Text-in-Chart': [],
        'Text-in-General': [],
        'Number-in-Chart': [],
        'Number-in-General': [],
    },

    'By Source': {
        'GPT-Sourced': [],
        'GPT-Inspired': [],
        'Completely Human': [],
    },

    'By Subject': {
        'Computer Science': [],
        'Economics': [],
        'Electrical Engineering and Systems Science': [],
        'Mathematics': [],
        'Physics': [],
        'Quantitative Biology': [],
        'Quantitative Finance': [],
        'Statistics': [],
    },

    'By Year': {
        '2020': [],
        '2021': [],
        '2022': [],
        '2023': [],
    },

    'By Subplot': {
        '1 Subplot': [], 
        '2-4 Subplots': [], 
        '5+ Subplots': [], 
    },
    
    'N_valid': [], 
    'N_invalid': [] 
}
