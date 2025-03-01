provider "aws" {
  region = var.region
}

# EC2 instance launched from custom AMI
resource "aws_instance" "my_ec2" {
  ami                     = var.ami_id
  instance_type           = var.instance_type
  key_name                = var.ssh_pem_file_name
  vpc_security_group_ids  = [var.security_group_id]
  
  tags = {
    Name = var.instance_name
  }
}

# Create an Ansible inventory file dynamically
resource "local_file" "ansible_inventory" {
  filename = "${path.module}/inventory.ini"  # Ensure file is created in the module directory
  content  = <<-EOF
  [my_ec2]
  ${aws_instance.my_ec2.public_ip} ansible_user=ubuntu ansible_ssh_private_key_file=${var.ssh_pem_file_name}.pem
  EOF
}

# Create an Ansible inventory file dynamically
resource "local_file" "backend_ip_route" {
  filename = "${path.module}/ansible/templates/api_route.js"  # Ensure file is created in the module directory
  content  = <<-EOF
  export const API_BASE_URL = "http://${aws_instance.my_ec2.public_ip}:${var.backend_port}";
  EOF
}

# Run Ansible playbooks
resource "null_resource" "run_ansible_playbook" {
  provisioner "local-exec" {
    command = "sleep 30 && ansible-playbook -i ${local_file.ansible_inventory.filename} ansible/playbook.yml"
    environment = {
      ANSIBLE_HOST_KEY_CHECKING = "False"
    }
  }

  depends_on = [
    aws_instance.my_ec2,
    local_file.ansible_inventory
  ]
}