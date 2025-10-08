from IPython.display import display, Markdown

def analyse(data):
    print("\n================================ Head ============================\n")
    display(data.head())
    print("\n=============================== Information ============================\n")
    display(data.info())
    print("\n============================= Non Numerical Description ============================\n")
    display(data.describe(include='O'))
    print("\n============================= Numerical Description ============================\n")
    display(data.describe())
