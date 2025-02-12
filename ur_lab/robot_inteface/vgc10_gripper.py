import pycurl
import xmlrpc.client
from io import BytesIO


class VG10C():
    def __init__(self, robot_ip: str, id: int):
        self.robot_ip = robot_ip
        self.id = id

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

    def vg10c_grip(self, channel=2, vacuum_power=60.):
        """
        channelA - 0
        channelB - 1
        both - 2
        power is integer from 0 to 80
        """

        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
        <methodName>vg10_grip</methodName>
            <params>
                <param>
                    <value><int>{self.id}</int></value>
                </param>
                <param>
                    <value><int>{channel}</int></value>
                </param>
                <param>
                    <value><double>{vacuum_power}</double></value>
                </param>
            </params>
        </methodCall>"""

        # if status != 0, then command not succesful. Perhaps there is no space to move the gripper
        return int(self._send_xml_rpc_request(xml_request))


    def vg10c_release(self, channel=2):
        """
        channelA - 0
        channelB - 1
        both - 2
        """
        release_channelA = int(channel == 0 or channel == 2)
        release_channelB = int(channel == 1 or channel == 2)

        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
        <methodName>vg10_release</methodName>
            <params>
                <param>
                    <value><int>{self.id}</int></value>
                </param>
                <param>
                    <value><boolean>{release_channelA}</boolean></value>
                </param>
                <param>
                    <value><boolean>{release_channelB}</boolean></value>
                </param>

            </params>
        </methodCall>"""

        # if status != 0, then command not succesful. Perhaps there is no space to move the gripper
        return int(self._send_xml_rpc_request(xml_request))

    def vg10c_get_vacuum(self,):
        # TODO set timeout and alert to understand if success?
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
        <methodName>vg10_get_vacuum</methodName>
            <params>
                <param>
                    <value><int>{self.id}</int></value>
                </param>
            </params>
        </methodCall>"""

        return self._send_xml_rpc_request(xml_request)

    def get_available_methods(self):
        """Query the server for available XML-RPC methods"""
        xml_request = """<?xml version="1.0"?>
        <methodCall>
        <methodName>system.listMethods</methodName>
        <params></params>
        </methodCall>"""

        try:
            methods = self._send_xml_rpc_request(xml_request)
            print("\nAvailable server methods:")
            for method in methods:
                print(f"- {method}")
            return methods
        except Exception as e:
            print(f"Could not retrieve methods list: {str(e)}")
            return None

    def get_method_help(self, method_name):
        """Get documentation for a specific method"""
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
        <methodName>system.methodHelp</methodName>
        <params>
            <param>
                <value><string>{method_name}</string></value>
            </param>
        </params>
        </methodCall>"""

        try:
            help_text = self._send_xml_rpc_request(xml_request)
            print(f"\nDocumentation for {method_name}:")
            print(help_text)
            return help_text
        except Exception as e:
            print(f"Could not retrieve help for {method_name}: {str(e)}")
            return None

    def get_method_signature(self, method_name):
        """Get parameter signature for a specific method"""
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
        <methodName>system.methodSignature</methodName>
        <params>
            <param>
                <value><string>{method_name}</string></value>
            </param>
        </params>
        </methodCall>"""

        try:
            signature = self._send_xml_rpc_request(xml_request)
            print(f"\nSignature for {method_name}:")
            print(signature)
            return signature
        except Exception as e:
            print(f"Could not retrieve signature for {method_name}: {str(e)}")
            return None


def main():
    # Default id is zero, if you have multiple grippers,
    # see logs in UR Teach Pendant to know which is which :)
    import time
    rg_id = 0
    ip = "192.168.0.10"
    gripper = VG10C(ip, rg_id)


    # gripper.get_available_methods()
    """
    - vg10_grip
    - vg10_release
    - vg10_idle
    - vg10_get_vacuum
    """

    # gripper.get_method_help("vg10_get_vacuum")
    # #
    # # # Get parameter signature for a specific method
    time.sleep(7)
    # gripper.get_method_signature("vg10_get_vacuum")


    gripper.vg10c_grip(channel=2, vacuum_power=80.)
    for i in range(100):
        print(gripper.vg10c_get_vacuum())
        time.sleep(0.1)

    print(gripper.vg10c_get_vacuum())
    gripper.vg10c_release(channel=2)


    ## TODO: figure out parameters, I don't think channels are correct
    ## TODO: test pickup
    ## TODO: clean code
    ## TODO: implement manipulator with gripper
    ##    Suction is 0-80


if __name__ == "__main__":
    main()
