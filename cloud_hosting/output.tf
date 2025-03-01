output "instance_public_ip" {
  description = "Public IP of the instance"
  value       = aws_instance.my_ec2.public_ip
}