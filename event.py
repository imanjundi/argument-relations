class Event:

    def __init__(self, node_dict):
        """
        The event object. Has the following attributes
        time: the time stamp as a number, eg. 3762717694
        action: the action taken (e.g. COMMENTED, RATED, REJECTED-SUGGESTION)
        attributes: can be None, can be a key, value pair with key
        attribute_key: can be None, otherwise describes the 'attribute type' that was affected by the action, e.g. the Name, description, username
        attribute_values: can be None, otherwise the value of the action, so if something was edited this contains new value of the element
        :param node_dict: the json dict of the event
        """
        self._time = node_dict["time"]
        self._action = node_dict["action"]
        self._attributes = node_dict["attributes"]
        self._attribute_key = self.get_attribute_key()
        self._attribute_value = self.get_attribute_value()

    def get_attribute_key(self):
        """Returns the attribute key if available, otherwise None"""
        if self._attributes:
            elements = self._attributes.split(" ")
            return elements[0].strip()
        return None

    def get_attribute_value(self):
        """Returns the attribute value if available, otherwise None"""
        if self._attributes:
            elements = self._attributes.split(" ")
            if len(elements) > 1:
                return elements[1].strip()
        return None

    def __str__(self):
        if self._attributes:
            s = "time: %s\naction: %s\nattributes: %s" % (self._time, self._action, self._attributes)
        else:
            s = "time: %s\naction: %s\nattributes: %s" % (self._time, self._action, "None")
        return s
