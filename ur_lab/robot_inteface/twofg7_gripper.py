"""
copied from this pull request that was not merged yet:
https://github.com/RyanPaulMcKenna/onRobot/blob/b385867f88417c08fda7dbb555469b75edf362e6/onRobot/onRobot/gripper.py
and slightly edited
"""


import pycurl
import xmlrpc.client
from io import BytesIO


class TwoFG7():
    def __init__(self, robot_ip: str, id: int):
        self.robot_ip = robot_ip
        self.id = id

        self.max_force = self.twofg_get_max_force()
        self.max_ext_width = self.twofg_get_max_external_width()
        self.max_int_width = self.twofg_get_max_internal_width()
        self.min_ext_width = self.twofg_get_min_external_width()
        self.min_int_width = self.twofg_get_min_internal_width()

        self.gripper_width = [self.twofg_get_external_width(), self.twofg_get_internal_width()]

    def _send_xml_rpc_request(self, _req=None):

        headers = ["Content-Type: application/x-www-form-urlencoded"]

        data = _req.replace('\r\n', '').encode()

        # Create a new cURL object
        curl = pycurl.Curl()

        # Set the URL to fetch
        curl.setopt(curl.URL, f'http://{self.robot_ip}:41414')
        curl.setopt(curl.HTTPHEADER, headers)
        curl.setopt(curl.POSTFIELDS, data)
        # Create a BytesIO object to store the response
        buffer = BytesIO()
        curl.setopt(curl.WRITEDATA, buffer)

        # Perform the request
        curl.perform()

        # Get the response body
        response = buffer.getvalue()

        # Print the response
        # print(response.decode('utf-8'))

        # Close the cURL object
        curl.close()
        # Get response from xmlrpc server
        xml_response = xmlrpc.client.loads(response.decode('utf-8'))

        return xml_response[0][0]

    def twofg_get_external_width(self) -> float:
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>twofg_get_external_width</methodName>
                <params>
                    <param>
                        <value><int>{self.id}</int></value>
                    </param>
                </params>
        </methodCall>"""

        return float(self._send_xml_rpc_request(xml_request))

    def twofg_get_internal_width(self) -> float:
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>twofg_get_internal_width</methodName>
                <params>
                    <param>
                        <value><int>{self.id}</int></value>
                    </param>
                </params>
        </methodCall>"""

        return float(self._send_xml_rpc_request(xml_request))

    def twofg_grip_external(self, target_width: float = 40.00, target_force: int = 20, speed: int = 1) -> int:
        """
            speed is the range from 1 to 100. It represents the percentage of the maximum speed.
        """
        assert target_width <= self.max_ext_width and target_width >= self.min_ext_width, f'Target Width must be within the range [{self.min_ext_width},{self.max_ext_width}]'
        assert target_force <= self.max_force or target_force >= 20, f'Target force must be within the range [20,{self.max_force}]'

        # WARNING: params will be sent straight to electrical system with no error checking on robot!
        if (target_width > self.max_ext_width):
            target_width = self.max_ext_width
        if (target_width < self.min_ext_width):
            target_width = self.min_ext_width
        if (target_force > self.max_force):
            target_force = self.max_force
        if (target_force < 20):
            target_force = 20

        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
        <methodName>twofg_grip_external</methodName>
            <params>
                <param>
                    <value><int>{self.id}</int></value>
                </param>
                <param>
                    <value><double>{target_width}</double></value>
                </param>
                <param>
                    <value><int>{target_force}</int></value>
                </param>
                <param>
                    <value><int>{speed}</int></value>
                </param>
            </params>
        </methodCall>"""

        # if status != 0, then command not succesful. Perhaps there is no space to move the gripper
        return int(self._send_xml_rpc_request(xml_request))

    def twofg_ext_release(self, target_width: float = 40.00, speed: int = 1) -> int:
        """
            speed is the range from 1 to 100. It represents the percentage of the maximum speed.
        """
        target_force: int = 80

        assert target_width <= self.max_ext_width and target_width >= self.min_ext_width, f'Target Width must be within the range [{self.min_ext_width},{self.max_ext_width}]'
        assert target_force <= self.max_force or target_force >= 20, f'Target force must be within the range [20,{self.max_force}]'

        # WARNING: params will be sent straight to electrical system with no error checking on robot!
        if (target_width > self.max_ext_width):
            target_width = self.max_ext_width
        if (target_width < self.min_ext_width):
            target_width = self.min_ext_width
        if (target_force > self.max_force):
            target_force = self.max_force
        if (target_force < 20):
            target_force = 20

        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
        <methodName>twofg_grip_external</methodName>
            <params>
                <param>
                    <value><int>{self.id}</int></value>
                </param>
                <param>
                    <value><double>{target_width}</double></value>
                </param>
                <param>
                    <value><int>{target_force}</int></value>
                </param>
                <param>
                    <value><int>{speed}</int></value>
                </param>
            </params>
        </methodCall>"""

        # if status != 0, then command not succesful. Perhaps there is no space to move the gripper
        return int(self._send_xml_rpc_request(xml_request))

    def twofg_grip_internal(self, target_width: float = 40.00, target_force: int = 10, speed: int = 1) -> int:
        """
            speed is the range from 1 to 100. It represents the percentage of the maximum speed.
        """
        assert target_width <= self.max_int_width and target_width >= self.min_int_width, f'Target Width must be within the range [{self.min_int_width},{self.max_int_width}]'
        assert target_force <= self.max_force or target_force >= 20, f'Target force must be within the range [20,{self.max_force}]'

        # WARNING: params will be sent straight to electrical system with no error checking on robot!
        if (target_width > self.max_int_width):
            target_width = self.max_int_width
        if (target_width < self.min_int_width):
            target_width = self.min_int_width
        if (target_force > self.max_force):
            target_force = self.max_force
        if (target_force < 20):
            target_force = 20

        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
        <methodName>twofg_grip_internal</methodName>
            <params>
                <param>
                    <value><int>{self.id}</int></value>
                </param>
                <param>
                    <value><double>{target_width}</double></value>
                </param>
                <param>
                    <value><int>{target_force}</int></value>
                </param>
                <param>
                    <value><int>{speed}</int></value>
                </param>
            </params>
        </methodCall>"""

        # if status != 0, then command not succesful. Perhaps there is no space to move the gripper
        return int(self._send_xml_rpc_request(xml_request))

    def twofg_int_release(self, target_width: float = 40.00, speed: int = 100) -> int:
        """
            speed is the range from 1 to 100. It represents the percentage of the maximum speed.
        """
        target_force: int = 80 # I think this is actually passed as width... need to further debug if relevant

        assert target_width <= self.max_int_width and target_width >= self.min_int_width, f'Target Width must be within the range [{self.min_int_width},{self.max_int_width}]'
        assert target_force <= self.max_force or target_force >= 20, f'Target force must be within the range [20,{self.max_force}]'

        # WARNING: params will be sent straight to electrical system with no error checking on robot!
        if (target_width > self.max_int_width):
            target_width = self.max_int_width
        if (target_width < self.min_int_width):
            target_width = self.min_int_width
        if (target_force > self.max_force):
            target_force = self.max_force
        if (target_force < 20):
            target_force = 20

        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
        <methodName>twofg_grip_internal</methodName>
            <params>
                <param>
                    <value><int>{self.id}</int></value>
                </param>
                <param>
                    <value><double>{target_width}</double></value>
                </param>
                <param>
                    <value><int>{target_force}</int></value>
                </param>
                <param>
                    <value><int>{speed}</int></value>
                </param>
            </params>
        </methodCall>"""

        # if status != 0, then command not succesful. Perhaps there is no space to move the gripper
        return int(self._send_xml_rpc_request(xml_request))

    def twofg_get_max_external_width(self) -> float:
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>twofg_get_max_external_width</methodName>
                <params>
                    <param>
                        <value><int>{self.id}</int></value>
                    </param>
                </params>
        </methodCall>"""

        return float(self._send_xml_rpc_request(xml_request))

    def twofg_get_max_internal_width(self) -> float:
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>twofg_get_max_internal_width</methodName>
                <params>
                    <param>
                        <value><int>{self.id}</int></value>
                    </param>
                </params>
        </methodCall>"""

        return float(self._send_xml_rpc_request(xml_request))

    def twofg_get_min_external_width(self) -> float:
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>twofg_get_min_external_width</methodName>
                <params>
                    <param>
                        <value><int>{self.id}</int></value>
                    </param>
                </params>
        </methodCall>"""

        return float(self._send_xml_rpc_request(xml_request))

    def twofg_get_min_internal_width(self) -> float:
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>twofg_get_min_internal_width</methodName>
                <params>
                    <param>
                        <value><int>{self.id}</int></value>
                    </param>
                </params>
        </methodCall>"""

        return float(self._send_xml_rpc_request(xml_request))

    def twofg_get_max_force(self) -> int:
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>twofg_get_max_force</methodName>
                <params>
                    <param>
                        <value><int>{self.id}</int></value>
                    </param>
                </params>
        </methodCall>"""

        return int(self._send_xml_rpc_request(xml_request))

    def twofg_get_status(self) -> int:
        # The status codes are not fully clear.
        # sofar:
        # 0: no grip
        # 2: has gripped an object

        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>twofg_get_status</methodName>
                <params>
                    <param>
                        <value><int>{self.id}</int></value>
                    </param>
                </params>
        </methodCall>"""

        return int(self._send_xml_rpc_request(xml_request))

    def twofg_get_busy(self) -> bool:
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>twofg_get_busy</methodName>
                <params>
                    <param>
                        <value><int>{self.id}</int></value>
                    </param>
                </params>
        </methodCall>"""

        return bool(self._send_xml_rpc_request(xml_request))

    def twofg_get_grip_detected(self) -> bool:
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>twofg_get_grip_detected</methodName>
                <params>
                    <param>
                        <value><int>{self.id}</int></value>
                    </param>
                </params>
        </methodCall>"""

        return bool(self._send_xml_rpc_request(xml_request))


def main():
    # Default id is zero, if you have multiple grippers,
    # see logs in UR Teach Pendant to know which is which :)
    print("Main")
    rg_id = 0
    ip = "192.168.0.10"
    gripper = TwoFG7(ip, rg_id)

    print(f"External Width: {gripper.twofg_get_external_width()}")
    print(f"Internal Width: {gripper.twofg_get_internal_width()}")
    print(f"Max External Width: {gripper.twofg_get_max_external_width()}")
    print(f"Max Internal Width: {gripper.twofg_get_max_internal_width()}")
    print(f"Min External Width: {gripper.twofg_get_min_external_width()}")
    print(f"Min Internal Width: {gripper.twofg_get_min_internal_width()}")
    print(f"Max Force: {gripper.twofg_get_max_force()}")

    print(gripper.twofg_grip_external(35.0, 40, 25))



if __name__ == "__main__":
    main()
