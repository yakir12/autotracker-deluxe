import json
import os


# Small class to act as a pass-through object for param/config file interaction.
class ParamFilePassthrough():
    def __init__(self):
        self.__fname = 'params.json'

        # This feels overcomplicated just to store the previous project but
        # there's scope to add stuff.
        self.__valid_keys = ["project_directory",
                             "project_file",
                             "options.autotracker.track_point",
                             "options.autotracker.cv_backend",
                             "options.autotracker.bg_computation_method",
                             "options.autotracker.bg_sample_size",
                             "options.autotracker.track_interval",
                             "options.autotracker.remember_roi",
                             "options.video.directory",
                             "options.autocalibration.fix_k1",
                             "options.autocalibration.fix_k2",
                             "options.autocalibration.fix_k3",
                             "options.autocalibration.fix_tangential",
                             "options.autocalibration.show_meta_text",
                             "options.processing.plot_grid",
                             "options.processing.include_legend",
                             "options.processing.filename",
                             "options.processing.filetype",
                             "options.processing.zero"
                             ]
        
        # Set some sane defaults
        self.__defaults = dict.fromkeys(self.__valid_keys)
        for k in self.__defaults.keys():
            self.__defaults[k] = ""

        self.__defaults["options.video.directory"] = "."

        self.__defaults["options.autotracker.track_point"] = "centre-of-mass"
        self.__defaults["options.autotracker.cv_backend"] = "BOOSTING"
        self.__defaults["options.autotracker.bg_computation_method"] = "first_N_median"
        self.__defaults["options.autotracker.bg_sample_size"] = 10
        self.__defaults["options.autotracker.track_interval"] = 1
        self.__defaults["options.autotracker.remember_roi"] = False

        self.__defaults["options.autocalibration.fix_k1"] = False
        self.__defaults["options.autocalibration.fix_k2"] = True
        self.__defaults["options.autocalibration.fix_k3"] = True
        self.__defaults["options.autocalibration.fix_tangential"] = True
        self.__defaults["options.autocalibration.show_meta_text"] = True

        self.__defaults["options.processing.plot_grid"] = True
        self.__defaults["options.processing.include_legend"] = True
        self.__defaults["options.processing.filename"] = "processed_tracks"
        self.__defaults["options.processing.filetype"] = "pdf"
        self.__defaults["options.processing.zero"] = False
        
        

        # If parameter file does not exist, create it as an emtpy json file.
        if not os.path.exists(self.__fname):
            with open(self.__fname, "w") as f:
                params = dict.fromkeys(self.__valid_keys)
                params = self.__set_defaults(params)

                # Set correct defaults
                for k in self.__valid_keys:
                    params[k] = self.__defaults[k]

                json.dump(params, f, indent=2)
        else:
            # If the file does exist, check that no elements are null.
            with open(self.__fname, "r") as f:
                params = json.load(f)
                
                for k in self.__valid_keys:
                    # Check all valid keys for sensible values
                    try:
                        if params[k] == None:
                            params[k] = self.__defaults[k]
                    except KeyError:
                        # If there is a valid entry which currently isn't in the
                        # file, then add it.
                        params[k] = self.__defaults[k]
                
            with open(self.__fname, "w") as f:
                json.dump(params, f, indent=2)

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
            json.dump(params, f, indent=2)

    def __set_defaults(self, params):
        for k in self.__valid_keys:
            params[k] = ""

        return params
   
# Insantiation of parameter file interaction object.
dtrack_params = ParamFilePassthrough()