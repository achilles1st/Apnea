import sys
import datetime
import csv
import argparse
import threading
from tkinter import messagebox
import serial
from dateutil import parser as dateparser
import numpy as np


class DataPoint():
    datatype = ''
    specs = {}  # {package_type: package_length, ...}
    attributes = []  # [(attribute, string, csvheader), ...]

    def __init__(self, package_type, package, time=False):
        self.time = time and time or datetime.datetime.now()

        # package type
        if package_type not in self.specs:
            raise ValueError("Invalid package type.")
        self.package_type = package_type

        # packet length
        if len(package) != self.specs[self.package_type]:
            raise ValueError("Invalid package length.")

        # set data
        self.set_package(package_type, package, time)

    def __repr__(self):
        hexBytes = ['0x{0:02X}'.format(byte) for byte in self.get_package()]
        return "{}({}, [{}], {})".format(
            self.__class__.__name__, self.package_type, ', '.join(hexBytes),
            repr(self.time))

    def __str__(self):
        return ",\n".join([attr[1] for attr in self.attributes]).format(
            **[getattr(self, attr[0]) for attr in self.attributes]
        )

    @classmethod
    def get_attribute_names(cls):
        return [attr[0] for attr in cls.attributes]

    @classmethod
    def get_csv_header(cls):
        return [attr[2] for attr in cls.attributes]

    def get_csv_data(self):
        return [getattr(self, attr[0]) for attr in self.attributes]

    def set_csv_data(self, data):
        for attr, _, key in self.attributes:
            if key in data:
                value = data[key]
                if isinstance(value, float):
                    value = int(value)
                if attr == 'time':
                    value = dateparser.parse(value)
                setattr(self, attr, value)

    def get_dict_data(self):
        ret = dict()
        for n, d in zip(self.get_csv_header(), self.get_csv_data()):
            ret[n] = d
        return ret

    def set_package(package_type, package, time):
        raise NotImplementedError('set_package() not implemented.')

    def get_package(self):
        raise NotImplementedError('get_package() not implemented.')


class RealtimeDataPoint(DataPoint):
    datatype = 'realtime'
    specs = {  # {package_type: package_length, ...}
        0x01: 7
    }
    attributes = [  # [(attribute, string, csvheader), ...]
        ('time',               "Time = {}",               "Time"),
        ('spO2',               "SpO2 = {}%",              "SpO2"),
        ('pulse_rate',         "Pulse Rate = {} bpm",     "PulseRate"),
        ('pulse_waveform',     "Pulse Waveform = {}",     "PulseWaveform"),
        ('pulse_beep',         "Pulse Beep = {}",         "PulseBeep"),
        ('bar_graph',          "Bar Graph = {}",          "BarGraph"),
        ('pi',                 "PI = {}%",                "Pi"),
        ('signal_strength',    "Signal Strength = {}",    "SignalStrength"),
        ('probe_error',        "Probe Error = {}",        "ProbeError"),
        ('low_spO2',           "Low SpO2 = {}",           "LowSpO2"),
        ('searching_too_long', "Searching Too Long = {}", "SearchingTooLong"),
        ('searching_pulse',    "Searching Pulse = {}",    "SearchingPulse"),
        ('spO2_invalid',       "SpO2 Invalid = {}",       "SpO2Invalid"),
        ('pulse_rate_invalid', "Pulse Rate Invalid = {}", "PulseRateInvalid"),
        ('pi_valid',           "PI Valid = {}",           "PiValid"),
        ('pi_invalid',         "PI Invalid = {}",         "PiInvalid"),
        ('reserved',           "Reserved = {}",           "Reserved"),
        ('datatype',           "Data Type = {}",          "DataType"),
        ('package_type',       "Package Type = {}",       "PackageType"),
    ]

    def set_package(self, package_type, package, time):
        # packet byte 2 / package byte 0
        self.signal_strength = package[0] & 0x0f
        self.searching_too_long = (package[0] & 0x10) >> 4
        self.low_spO2 = (package[0] & 0x20) >> 5
        self.pulse_beep = (package[0] & 0x40) >> 6
        self.probe_error = (package[0] & 0x80) >> 7

        # packet byte 3 / package byte 1
        self.pulse_waveform = package[1] & 0x7f
        self.searching_pulse = (package[1] & 0x80) >> 7

        # packet byte 4 / package byte 2
        self.bar_graph = package[2] & 0x0f
        self.pi_valid = (package[2] & 0x10) >> 4
        self.reserved = (package[2] & 0xe0) >> 5

        # packet byte 5 / package byte 3
        self.pulse_rate = package[3]
        self.pulse_rate_invalid = int(self.pulse_rate == 0xff)

        # packet byte 6 / package byte 4
        self.spO2 = package[4]
        self.spO2_invalid = int(self.spO2 == 0x7f)

        # packet byte 7-8 / package byte 5-6
        self.pi = package[6] << 8 | package[5]
        self.pi_invalid = int(self.pi == 0xffff)

    def get_package(self):
        package = [0] * self.specs[self.package_type]

        # packet byte 2 / package byte 0
        package[0] = self.signal_strength & 0x0f
        if self.searching_too_long:
            package[0] |= 0x10
        if self.low_spO2:
            package[0] |= 0x20
        if self.pulse_beep:
            package[0] |= 0x40
        if self.probe_error:
            package[0] |= 0x80

        # packet byte 3 / package byte 1
        package[1] = self.pulse_waveform & 0x7f
        if self.searching_pulse:
            package[1] |= 0x80

        # packet byte 4 / package byte 2
        package[2] = self.bar_graph & 0x0f
        if self.pi_valid:
            package[2] |= 0x10
        package[2] |= (self.reserved << 5) & 0xe0

        # packet byte 5 / package byte 3
        package[3] = self.pulse_rate & 0xff

        # packet byte 6 / package byte 4
        package[4] = self.spO2 & 0xff

        # packet byte 7-8 / package byte 5-6
        package[5] = self.pi & 0x00ff
        package[6] = (self.pi & 0xff00) >> 8

        return package


class StorageDataPoint(DataPoint):
    datatype = 'storage'
    specs = {  # {package_type: package_length, ...}
        0x0f: 2,  # one package of 6 bytes split into 3 datapoints
        0x09: 4,
    }
    attributes = [  # [(attribute, string, csvheader), ...]
        ('time',               "Time = {}",               "Time"),
        ('spO2',               "SpO2 = {}%",              "SpO2"),
        ('pulse_rate',         "Pulse Rate = {} bpm",     "PulseRate"),
        ('pi',                 "PI = {}%",                "Pi"),
        ('pi_support',         "PI Support = {}",         "PiSupport"),
        ('pulse_rate_invalid', "Pulse Rate Invalid = {}", "PulseRateInvalid"),
        ('spO2_invalid',       "SpO2 Invalid = {}",       "SpO2Invalid"),
        ('pi_invalid',         "PI Invalid = {}",         "PiInvalid"),
        ('datatype',           "Data Type = {}",          "DataType"),
        ('package_type',       "Package Type = {}",       "PackageType"),
    ]

    def set_package(self, package_type, package, time):
        # pi support
        self.pi_support = 0
        if self.package_type == 0x09:
            self.pi_support = 1

        # packet byte 2|4|6 / package byte 0
        self.spO2 = package[0] & 0xff
        self.spO2_invalid = int(self.spO2 == 0x7f)

        # packet byte 3|5|7 / package byte 1
        self.pulse_rate = package[1] & 0xff
        self.pulse_rate_invalid = int(self.pulse_rate == 0xff)

        # packet byte 4-5 / package byte 2-3
        if self.pi_support:
            self.pi = package[3] << 8 | package[2]
            self.pi_invalid = int(self.pi == 0xffff)
        else:
            self.pi = "-"
            self.pi_invalid = "-"

    def get_package(self):
        package = [0] * self.specs[self.package_type]

        # packet byte 2|4|6 / package byte 0
        package[0] = self.spO2 & 0xff

        # packet byte 3|5|7 / package byte 1
        package[1] = self.pulse_rate & 0xff

        # packet byte 4-5 / package byte 2-3
        if self.pi_support:
            package[2] = self.pi & 0x00ff
            package[3] = (self.pi & 0xff00) >> 8

        return package


class CMS50Dplus():
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=0.5,
                 connect=True):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.keepalive_interval = datetime.timedelta(seconds=5)
        self.keepalive_timestamp = datetime.datetime.now()
        self.storage_time_interval = datetime.timedelta(seconds=1)
        self.connection = None
        if connect:
            self.connect()

    def __del__(self):
        self.disconnect()

    @staticmethod
    def set_bit(byte, value=1, index=7):
        mask = 1 << index
        byte &= ~mask
        if value:
            byte |= mask
        return byte

    @classmethod
    def decode_package(cls, packets):
        # check packet length
        if len(packets) < 3:
            raise ValueError("Package too short to decode.")
        if len(packets) > 9:
            raise ValueError("Package too long to decode")

        # check synchronization bits
        if packets[0] & 0x80:
            raise ValueError("Invalid synchronization bit.")
        for byte in packets[1:]:
            if not byte & 0x80:
                raise ValueError("Invalid synchronization bit.")

        # define packet parts
        package_type = packets[0]
        high_byte = packets[1]
        package = packets[2:]

        # decode high byte
        for idx, byte in enumerate(package):
            package[idx] = cls.set_bit(byte, high_byte & 0x01 << idx)

        return package_type, package

    @classmethod
    def encode_package(cls, package_type, package,
                       padding=0, padding_byte=0x00):
        # check package length
        if len(package) > 7:
            raise ValueError("Package too long to encode.")

        # define packet parts
        high_byte = 0x80
        package = package[:]

        # pad package
        if padding:
            if padding < len(package):
                raise ValueError("Padding too short.")
            if padding > 7:
                raise ValueError("Padding too long.")
            if padding > len(package):
                package += [padding_byte] * (padding - len(package))

        # encode high byte
        for idx, byte in enumerate(package):
            high_byte |= (byte & 0x80) >> (7 - idx)

        # set synchronization bits
        package_type = cls.set_bit(package_type, 0)
        for idx, byte in enumerate(package):
            package[idx] = cls.set_bit(byte)

        # compose packets
        packets = [package_type, high_byte] + package

        return packets

    def is_connected(self):
        if self.connection and self.connection.isOpen():
            return True
        return False

    def connect(self):
        if self.connection is None:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                xonxoff=1
            )
        elif not self.is_connected():
            self.connection.open()

    def disconnect(self):
        if self.is_connected():
            self.connection.close()

    def get_byte(self):
        char = self.connection.read()
        if len(char) == 0:
            return None
        else:
            return ord(char)

    def send_bytes(self, values):
        return self.connection.write(
            b''.join([
                chr(value & 0xff).encode('raw_unicode_escape')
                for value in values]))

    def expect_byte(self, value):
        while True:
            byte = self.get_byte()
            if byte is None:
                return False
            elif byte == value:
                return True

    def send_command(self, command, data=[]):
        package = self.encode_package(
            package_type=0x7d,  # command
            package=[command] + data, padding=7, padding_byte=0x00)
        self.send_bytes(package)
        self.connection.flush()

    def send_keepalive(self):
        now = datetime.datetime.now()
        if now - self.keepalive_timestamp > self.keepalive_interval:
            self.send_command(0xaf)  # keepalive
            self.keepalive_timestamp = now

    def get_packets(self, amount=0):
        count = 0
        idx = 0
        packets = []
        while True:
            if not amount:
                self.send_keepalive()
            byte = self.get_byte()
            if byte is None:
                if len(packets[:idx]) < 3:
                    raise ValueError("Recieved too few bytes for packets.")
                if amount and count + 1 < amount:
                    raise ValueError("Recieved too few packets.")
                yield packets[:idx]
                break
            sync_bit = bool(byte & 0x80)
            if not sync_bit:
                if packets:
                    if len(packets[:idx]) < 3:
                        raise ValueError("Recieved too few bytes for packets.")
                    yield packets[:idx]
                    if amount:
                        count += 1
                        if count == amount:
                            break
                packets = [0x00] * 9
                idx = 0
            if idx > 8:
                raise ValueError("Received too many bytes for packets.")
            packets[idx] = byte
            idx += 1

    def get_packages(self, amount=0):
        for packets in self.get_packets(amount):
            package_type, package = self.decode_package(packets)
            if package_type == 0x0d:  # disconnect notice
                if package[0] in [0x00, 0x01]:
                    break
                raise ValueError(
                    "Received reasoncode 0x{:02X}".format(package[0]))
            yield package_type, package

    def get_realtime_data(self):
        try:
            self.connection.reset_input_buffer()
            self.send_command(0xa1)  # start realtime data
            for package_type, package in self.get_packages():
                yield RealtimeDataPoint(package_type, package)
        except KeyboardInterrupt:
            pass
        finally:
            self.send_command(0xa2)  # stop realtime data

    def get_storage_data(self, starttime=False,
                         user_index=0x01, storage_segment=0x01):
        if not starttime:
            starttime = datetime.datetime.now()
        try:
            self.connection.reset_input_buffer()
            self.send_command(  # start storage data
                0xa6, [user_index, storage_segment])
            for package_type, package in self.get_packages():
                if package_type == 0x0f:
                    if package[0] and package[1]:
                        yield StorageDataPoint(
                            package_type, package[0:2], time=starttime)
                        starttime += self.storage_time_interval
                    if package[2] and package[3]:
                        yield StorageDataPoint(
                            package_type, package[2:4], time=starttime)
                        starttime += self.storage_time_interval
                    if package[4] and package[5]:
                        yield StorageDataPoint(
                            package_type, package[4:6], time=starttime)
                        starttime += self.storage_time_interval
                else:
                    yield StorageDataPoint(
                        package_type, package, time=starttime)
                    starttime += self.storage_time_interval
        except KeyboardInterrupt:
            pass
        finally:
            self.send_command(0xa7)  # stop storage data


class ThreadedRealtimeData(threading.Thread):

    def __init__(self, root, oximeter, data):
        threading.Thread.__init__(self)
        self.root = root
        self.oximeter = oximeter
        self.data = data

    def run(self):
        try:
            datapoints = self.oximeter.get_realtime_data()
            for datapoint in datapoints:

                # gracious thread end
                if getattr(self.root, 'stop_thread', False):
                    break

                # expand data if running out of allocated space
                if len(self.data['time']) == self.data['count']:
                    for key in self.data:
                        if isinstance(self.data[key], list):
                            self.data[key] += [0] * 100000

                # add data
                idx = self.data['count']
                self.data['point'][idx] = datapoint
                self.data['time'][idx] = datapoint.time

                spO2 = np.nan
                if datapoint.spO2:
                    spO2 = datapoint.spO2
                self.data['spO2'][idx] = spO2
                pulse_rate = np.nan
                if datapoint.pulse_rate:
                    pulse_rate = datapoint.pulse_rate
                self.data['pulse_rate'][idx] = pulse_rate
                pulse_waveform = np.nan
                if datapoint.pulse_waveform:
                    pulse_waveform = datapoint.pulse_waveform
                self.data['pulse_waveform'][idx] = pulse_waveform
                self.data['pulse_beep'][idx] = datapoint.pulse_beep
                self.data['bar_graph'][idx] = datapoint.bar_graph
                self.data['pi'][idx] = datapoint.pi
                self.data['signal_strength'][idx] = datapoint.signal_strength
                self.data['probe_error'][idx] = datapoint.probe_error
                self.data['low_spO2'][idx] = datapoint.low_spO2
                self.data[
                    'searching_too_long'][idx] = datapoint.searching_too_long
                self.data['searching_pulse'][idx] = datapoint.searching_pulse
                self.data['spO2_invalid'][idx] = datapoint.spO2_invalid
                self.data[
                    'pulse_rate_invalid'][idx] = datapoint.pulse_rate_invalid
                self.data['pi_valid'][idx] = datapoint.pi_valid
                self.data['pi_invalid'][idx] = datapoint.pi_invalid
                self.data['reserved'][idx] = datapoint.reserved

                # calculate samplerate
                if self.data['count'] > 1:
                    start = self.data['time'][0]
                    end = self.data['time'][self.data['count']]
                    seconds = (end - start).total_seconds()
                    self.data['samplerate'] = self.data['count'] / seconds

                # count datapoint
                self.data['count'] += 1

        except Exception as e:
            self.root.thread_exception = True
            messagebox.showerror(parent=self.root, title='Error:', message=e)


def print_realtime_data(port):
    print("Saving live data...")
    print("Press CTRL-C / disconnect the device to terminate data collection.")

    oximeter = CMS50Dplus(port)
    datapoints = oximeter.get_realtime_data()
    try:
        for datapoint in datapoints:
            sys.stdout.write(
                "\rSignal: {:>2}"
                " | PulseRate: {:>3}"
                " | PulseWave: {:>3}"
                " | SpO2: {:>2}%"
                " | ProbeError: {:>1}".format(
                    datapoint.signal_strength,
                    datapoint.pulse_rate,
                    datapoint.pulse_waveform,
                    datapoint.spO2,
                    datapoint.probe_error))
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass


def dump_realtime_data(port, filename):
    print("Saving live data...")
    print("Press CTRL-C / disconnect the device to terminate data collection.")


    oximeter = CMS50Dplus(port)
    datapoints = oximeter.get_realtime_data()
    measurements = 0
    try:
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(RealtimeDataPoint.get_csv_header())
            for datapoint in datapoints:
                writer.writerow(datapoint.get_csv_data())
                measurements += 1
                sys.stdout.write(
                    "\rGot {0} measurements...".format(measurements))
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass


def valid_datetime(s):
    try:
        return dateparser.parse(s)
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


if __name__ == "__main__":

    port = "COM8"
    filename = False

    # cli
    if not filename:
        print_realtime_data(port)
    else:
        dump_realtime_data(port, filename)

    print("\nDone.")

