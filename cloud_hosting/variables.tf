variable "region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "ami_id" {
  description = "AMI ID to launch the instance"
  type        = string
  default     = "ami-015d200881ec57abe"
}

variable "instance_type" {
  description = "AWS instance type"
  default     = "t2.micro"
}

variable "ssh_pem_file_name" {
  description = "File name of the SSH private key"
  type        = string
  default    = "shafeeq-thinkpad"
}

variable "security_group_id" {
  description = "Security group ID"
  type        = string
  default     = "sg-0b71b5574fab565c6"
}

variable "backend_port" {
  description = "Port on which the backend server is running"
  type        = number
  default     = 8000
}

variable "instance_name" {
  description = "Name of the EC2 instance to be launched"
  type        = string
  default     = "FMD-ML-Deployment-Instance"
}