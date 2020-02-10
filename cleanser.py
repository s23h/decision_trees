import os
import pandas as pd

class Cleanser():

    directory = None
    df = None

    def batch_cleanse(self):

    # directory = os.fsencode(PATH)
        for file in os.listdir(self.directory):
            filename = str(os.fsdecode(file))
            if filename.endswith(".csv"):
                self.cleanse(os.path.join(self.directory, filename))
            else:
                continue

    def cleanse(self, filename):
        df = pd.read_csv(filename)
        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        df = df.drop(cols_to_drop, axis=1)

        gf = df.groupby('Code')

        for name, group in gf:
            with open("out/" + name, 'a') as f:
                group.to_csv(f, header=f.tell()==0, index=False)



c = Cleanser()
d = c.batch_cleanse()
