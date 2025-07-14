
TOKEN="evOmQItfKbh2qjZ1vwuWlG7Q4EFJtxtx"
PORT=42332
ADDRESS=root@0z89le1ioatj3cyksnow.deepln.com

sshpass -p $TOKEN scp -rP $PORT * $ADDRESS:/data/coding/gomoku/