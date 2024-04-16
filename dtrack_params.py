import json
import os


# Small class to act as a pass-through object for param/config file interaction.
class ParamFilePassthrough():
    def __init__(self):
        self.__fname = 'params.json'

        # This feels overcomplicated just to store the previous project but
        # there's scope to add stuff.
        self.__valid_keys = ["project_directory",
                             "project_file"]

        # If parameter file does not exist, create it as an emtpy json file.
        if not os.path.exists(self.__fname):
            with open(self.__fname, "w") as f:
                params = dict.fromkeys(self.__valid_keys)
                #params = self.__set_defaults(params)
                print(params)
                json.dump(params, f)

    def __getitem__(self,key):
        with open(self.__fname, "r") as f:
                params = json.load(f)
        return params[key]
    
    def __setitem__(self, key, value):
        # Failing this assertion indicates programmer error.
        assert(key in self.__valid_keys)

        with open(self.__fname, "r") as f:
            params = json.load(f)

        params[key] = value

        with open(self.__fname, "w") as f:
            json.dump(params, f)

    def __set_defaults(self, params):
        for k in self.__valid_keys:
            params[k] = ""
   
# Insantiation of parameter file interaction object.
dtrack_params = ParamFilePassthrough()