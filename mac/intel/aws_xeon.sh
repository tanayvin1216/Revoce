#!/usr/bin/env bash
# aws_xeon.sh — rent a real Intel Xeon on AWS for ~15 minutes, run the Intel
# benchmark bundle on it, pull the results back, terminate it.
#
#   AWS_PROFILE=tanay bash mac/intel/aws_xeon.sh launch     # starts the instance (COSTS MONEY)
#   AWS_PROFILE=tanay bash mac/intel/aws_xeon.sh run        # rsync repo, run run_on_intel.sh, fetch results
#   AWS_PROFILE=tanay bash mac/intel/aws_xeon.sh terminate  # stop paying
#
# Instance: c7i.2xlarge = 4th-gen Intel Xeon Scalable (Sapphire Rapids, AMX),
# 8 vCPU / 16 GiB, ~$0.36/h on-demand in us-east-1. That is the "general-purpose
# Intel hardware (Xeon)" rubric line; it has no NPU, so Core Ultra stays unmeasured.
# The benchmark JSONs record the CPU brand string, so the provenance is in the data.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
REGION="${AWS_REGION:-us-east-1}"
TYPE="${INSTANCE_TYPE:-c7i.2xlarge}"
KEY_NAME="${KEY_NAME:-voicelink-bench}"
PUBKEY="${PUBKEY:-$HOME/.ssh/id_ed25519.pub}"
STATE="$HERE/artifacts/aws_xeon_instance.txt"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i ${PUBKEY%.pub}"

ami() {  # latest Ubuntu 22.04 x86_64 from Canonical's official SSM parameter
  aws ssm get-parameter --region "$REGION" \
    --name /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
    --query Parameter.Value --output text
}

case "${1:-}" in
  launch)
    aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1 || \
      aws ec2 import-key-pair --region "$REGION" --key-name "$KEY_NAME" \
        --public-key-material "fileb://$PUBKEY" >/dev/null
    SG=$(aws ec2 describe-security-groups --region "$REGION" --group-names voicelink-bench \
          --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
    if [ -z "$SG" ] || [ "$SG" = "None" ]; then
      SG=$(aws ec2 create-security-group --region "$REGION" --group-name voicelink-bench \
            --description "ssh only, voicelink benchmark" --query GroupId --output text)
      MYIP=$(curl -s https://checkip.amazonaws.com)
      aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
        --protocol tcp --port 22 --cidr "$MYIP/32" >/dev/null
    fi
    ID=$(aws ec2 run-instances --region "$REGION" --image-id "$(ami)" --instance-type "$TYPE" \
          --key-name "$KEY_NAME" --security-group-ids "$SG" \
          --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3"}}]' \
          --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=voicelink-bench}]' \
          --query 'Instances[0].InstanceId' --output text)
    echo "$ID" > "$STATE"
    echo "launched $ID ($TYPE); waiting for it to come up..."
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$ID"
    IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$ID" \
          --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    echo "$IP" >> "$STATE"
    echo "up at $IP — now: bash mac/intel/aws_xeon.sh run   (remember to terminate)"
    ;;
  run)
    IP=$(sed -n 2p "$STATE")
    until ssh $SSH_OPTS ubuntu@"$IP" true 2>/dev/null; do sleep 5; done
    ssh $SSH_OPTS ubuntu@"$IP" 'sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-venv rsync >/dev/null && lscpu | grep "Model name"'
    rsync -az -e "ssh $SSH_OPTS" --exclude .git --exclude '*.pt' --exclude 'minilm-ov' --exclude 'whisper-base.en-ov-int8' \
      --exclude __pycache__ --exclude .pytest_cache "$REPO"/ ubuntu@"$IP":~/VoiceLink/
    rsync -az -e "ssh $SSH_OPTS" "$REPO"/mac/evaluation/phase4/student.pt ubuntu@"$IP":~/VoiceLink/mac/evaluation/phase4/
    ssh $SSH_OPTS ubuntu@"$IP" 'cd ~/VoiceLink && python3 -m venv .venv && . .venv/bin/activate && pip install -q --upgrade pip && VOICELINK_PYTHON=python3 bash mac/intel/run_on_intel.sh' 2>&1 | tee "$HERE/artifacts/aws_xeon_run.log"
    rsync -az -e "ssh $SSH_OPTS" ubuntu@"$IP":~/VoiceLink/mac/intel/artifacts/'e2e_*.json' ubuntu@"$IP":~/VoiceLink/mac/intel/artifacts/'whisper_bench_*.json' "$HERE/artifacts/" || true
    rsync -az -e "ssh $SSH_OPTS" ubuntu@"$IP":~/VoiceLink/mac/intel/BENCHMARKS.md "$HERE/BENCHMARKS.xeon.md"
    echo "results in mac/intel/artifacts/ and mac/intel/BENCHMARKS.xeon.md — now TERMINATE."
    ;;
  terminate)
    ID=$(sed -n 1p "$STATE")
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
    ;;
  *) sed -n 2,12p "$0"; exit 2 ;;
esac
