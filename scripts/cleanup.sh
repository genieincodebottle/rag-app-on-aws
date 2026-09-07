#!/bin/bash

# ============================================================
# AWS RAG Application Cleanup Script
# ============================================================
# This script thoroughly removes all AWS resources created for the RAG application
# with proper dependency handling and verification

# Set text colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================"
echo " AWS RAG Application - Complete Resource Cleanup Script  "
echo "============================================================"
echo -e "${NC}"

# Get Project configuration from GitHub action env variables
PROJECT_NAME=${PROJECT_NAME}
STAGE=${STAGE}
AWS_REGION=${AWS_REGION}

# Standard prefix for resources
PREFIX="${PROJECT_NAME}-${STAGE}"

echo -e "\n${YELLOW}This script will DELETE ALL resources for:"
echo -e "  Project: ${PROJECT_NAME}"
echo -e "  Stage: ${STAGE}"
echo -e "  Region: ${AWS_REGION}${NC}"
echo -e "${RED}WARNING: This action is IRREVERSIBLE and will delete ALL data!${NC}"
echo -e "\n${YELLOW}Starting cleanup process...${NC}"

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${YELLOW}Warning: $1 command not found. Some functionality may be limited.${NC}"
        return 1
    fi
    return 0
}

# Check for AWS CLI
if ! check_command aws; then
    echo -e "${RED}Error: AWS CLI is required for this script.${NC}"
    exit 1
fi

# Check for jq command
if ! check_command jq; then
    echo -e "${YELLOW}Warning: jq command not found. Will use alternative methods for JSON processing.${NC}"
    USE_JQ=false
else
    USE_JQ=true
fi

# Function to run a command with error handling
run_command() {
    local cmd="$1"
    local error_msg="$2"
    local success_msg="$3"
    local ignore_error="${4:-false}"
    
    echo -e "${BLUE}Running: $cmd${NC}"
    
    # Execute the command
    eval "$cmd"
    local status=$?
    
    if [ $status -eq 0 ]; then
        if [ -n "$success_msg" ]; then
            echo -e "${GREEN}$success_msg${NC}"
        fi
        return 0
    else
        if [ "$ignore_error" = "true" ]; then
            if [ -n "$error_msg" ]; then
                echo -e "${YELLOW}$error_msg (continuing)${NC}"
            fi
            return 0
        else
            if [ -n "$error_msg" ]; then
                echo -e "${RED}$error_msg${NC}"
            fi
            return 1
        fi
    fi
}

# Function to wait with a countdown
wait_with_message() {
    local seconds="$1"
    local message="$2"
    
    echo -n -e "${YELLOW}$message "
    for (( i=seconds; i>=1; i-- )); do
        echo -n -e "$i... "
        sleep 1
    done
    echo -e "${NC}"
}

# Function to check if a resource exists and return an identifier
resource_exists() {
    local resource_type="$1"
    local identifier="$2"
    local query="$3"
    local exists=false
    local id=""
    
    case "$resource_type" in
        "lambda")
            # Check if lambda function exists
            id=$(aws lambda get-function --function-name "$identifier" --region $AWS_REGION --query "Configuration.FunctionName" --output text 2>/dev/null || echo "")
            if [ -n "$id" ] && [ "$id" != "None" ]; then exists=true; fi
            ;;
        "api")
            # Check if API Gateway exists
            id=$(aws apigateway get-rest-apis --region $AWS_REGION --query "items[?name=='$identifier'].id" --output text 2>/dev/null || echo "")
            if [ -n "$id" ] && [ "$id" != "None" ]; then exists=true; fi
            ;;
        "cognito")
            # Check if Cognito User Pool exists
            id=$(aws cognito-idp list-user-pools --max-results 20 --region $AWS_REGION --query "UserPools[?Name=='$identifier'].Id" --output text 2>/dev/null || echo "")
            if [ -n "$id" ] && [ "$id" != "None" ]; then exists=true; fi
            ;;
        "bucket")
            # Check if S3 bucket exists - fixed to return just the bucket name
            if aws s3api head-bucket --bucket "$identifier" --region $AWS_REGION 2>/dev/null; then
                id="$identifier"  # Return just the bucket name
                exists=true
            fi
            ;;
        "table")
            # Check if DynamoDB table exists
            id=$(aws dynamodb describe-table --table-name "$identifier" --region $AWS_REGION --query "Table.TableName" --output text 2>/dev/null || echo "")
            if [ -n "$id" ] && [ "$id" != "None" ]; then exists=true; fi
            ;;
        "rds")
            # Check if RDS instance exists
            id=$(aws rds describe-db-instances --db-instance-identifier "$identifier" --region $AWS_REGION --query "DBInstances[0].DBInstanceIdentifier" --output text 2>/dev/null || echo "")
            if [ -n "$id" ] && [ "$id" != "None" ]; then exists=true; fi
            ;;
        "vpc")
            # Check if VPC exists
            id=$(aws ec2 describe-vpcs --region $AWS_REGION --filters "$query" --query "Vpcs[0].VpcId" --output text 2>/dev/null || echo "")
            if [ -n "$id" ] && [ "$id" != "None" ]; then exists=true; fi
            ;;
        "secret")
            # Check if Secret exists
            id=$(aws secretsmanager describe-secret --secret-id "$identifier" --region $AWS_REGION --query "ARN" --output text 2>/dev/null || echo "")
            if [ -n "$id" ] && [ "$id" != "None" ]; then exists=true; fi
            ;;
        "log-group")
            # Check if CloudWatch Log Group exists
            id=$(aws logs describe-log-groups --log-group-name-prefix "$identifier" --region $AWS_REGION --query "logGroups[0].logGroupName" --output text 2>/dev/null || echo "")
            if [ -n "$id" ] && [ "$id" != "None" ]; then exists=true; fi
            ;;
        *)
            echo "Unknown resource type: $resource_type"
            ;;
    esac
    
    if [ "$exists" = true ]; then
        echo "$id"
        return 0
    else
        return 1
    fi
}

# Function to handle Lambda functions with VPC access
handle_lambda_vpc_dependencies() {
    local vpc_id="$1"
    
    echo -e "\n${YELLOW}Identifying Lambda functions with VPC access...${NC}"
    
    # Find all Lambda functions with VPC configuration
    local vpc_lambdas=$(aws lambda list-functions --region $AWS_REGION --query "Functions[?VpcConfig.VpcId=='$vpc_id'].FunctionName" --output text)
    
    if [ -n "$vpc_lambdas" ]; then
        echo "Found Lambda functions with VPC access: $vpc_lambdas"
        
        for func in $vpc_lambdas; do
            echo "Updating Lambda function $func to remove VPC configuration..."
            
            # Update the function to remove VPC configuration
            run_command "aws lambda update-function-configuration --function-name $func --vpc-config SubnetIds=[],SecurityGroupIds=[] --region $AWS_REGION" \
                "Failed to remove VPC configuration from Lambda function: $func" \
                "Successfully removed VPC configuration from Lambda function: $func" \
                "true"
        done
        
        # Wait for the Lambda ENIs to be deleted
        echo "Waiting for Lambda ENIs to be deleted..."
        wait_with_message 60 "Waiting for Lambda ENIs to be cleaned up..."
    else
        echo "No Lambda functions with VPC access found."
    fi
}

# Function to delete vpc
delete_vpc() {
    local vpc_id="$1"
    local has_dependencies=false
    
    echo -e "\n${YELLOW}Starting streamlined VPC deletion process for VPC: $vpc_id${NC}"
    
    # First handle Lambda VPC dependencies
    handle_lambda_vpc_dependencies "$vpc_id"
    
    # 1. Terminate EC2 instances
    echo "Checking for EC2 instances..."
    instances=$(aws ec2 describe-instances --filters "Name=vpc-id,Values=$vpc_id" "Name=instance-state-name,Values=pending,running,stopping,stopped" --query "Reservations[].Instances[].InstanceId" --output text --region $AWS_REGION)
    
    if [ -n "$instances" ]; then
        echo "Found EC2 instances in the VPC. Terminating: $instances"
        run_command "aws ec2 terminate-instances --instance-ids $instances --region $AWS_REGION" \
            "Failed to terminate EC2 instances" \
            "Successfully initiated termination of EC2 instances" \
            "true"
        
        echo "Waiting for instances to terminate..."
        run_command "aws ec2 wait instance-terminated --instance-ids $instances --region $AWS_REGION" \
            "Failed to wait for instance termination" \
            "All instances terminated" \
            "true"
    fi
    
    # 2. Delete load balancers
    echo "Checking for load balancers..."
    lb_arns=$(aws elbv2 describe-load-balancers --region $AWS_REGION --query "LoadBalancers[?VpcId=='$vpc_id'].LoadBalancerArn" --output text)
    
    if [ -n "$lb_arns" ]; then
        for lb_arn in $lb_arns; do
            echo "Deleting load balancer: $lb_arn"
            run_command "aws elbv2 delete-load-balancer --load-balancer-arn $lb_arn --region $AWS_REGION" \
                "Failed to delete load balancer" \
                "Successfully deleted load balancer" \
                "true"
        done
        wait_with_message 15 "Waiting for load balancers to be deleted..."
    fi
    
    # 3. Delete NAT Gateways
    echo "Checking for Nat Gateways..."
    nat_gateways=$(aws ec2 describe-nat-gateways --filter "Name=vpc-id,Values=$vpc_id" --query "NatGateways[?State!='deleted'].NatGatewayId" --output text --region $AWS_REGION)
    
    if [ -n "$nat_gateways" ]; then
        for nat_id in $nat_gateways; do
            echo "Deleting NAT Gateway: $nat_id"
            run_command "aws ec2 delete-nat-gateway --nat-gateway-id $nat_id --region $AWS_REGION" \
                "Failed to delete NAT Gateway: $nat_id" \
                "Successfully initiated deletion of NAT Gateway: $nat_id" \
                "true"
        done
        
        # Wait for NAT gateways to be deleted
        echo "Waiting for NAT gateways to be deleted..."
        wait_with_message 45 "Waiting for NAT gateway deletion..."
    fi
    
    # 4. Release Elastic IPs
    echo "Finding and releasing Elastic IPs associated with VPC resources..."
    enis=$(aws ec2 describe-network-interfaces --filters "Name=vpc-id,Values=$vpc_id" --query "NetworkInterfaces[].NetworkInterfaceId" --output text --region $AWS_REGION)
    
    for eni in $enis; do
        eips=$(aws ec2 describe-addresses --filters "Name=network-interface-id,Values=$eni" --query "Addresses[].AllocationId" --output text --region $AWS_REGION)
        
        for eip in $eips; do
            if [ -n "$eip" ]; then
                echo "Releasing Elastic IP: $eip"
                run_command "aws ec2 release-address --allocation-id $eip --region $AWS_REGION" \
                    "Failed to release Elastic IP" \
                    "Successfully released Elastic IP" \
                    "true"
            fi
        done
    done

    # 4. Release Elastic IPs based on Project Prefix
    echo "Finding and releasing remaining Elastic IPs associated with project prefix..."
    enis=$(aws ec2 describe-addresses --filters "Name=tag:Name,Values=${PREFIX}-*" --query 'Addresses[].{Name:Tags[?Key==`Name`]|[0].Value, PublicIp:PublicIp, AllocationId:AllocationId}' --output table --region $AWS_REGION)
    
    for eni in $enis; do
        eips=$(aws ec2 describe-addresses --filters "Name=network-interface-id,Values=$eni" --query "Addresses[].AllocationId" --output text --region $AWS_REGION)
        
        for eip in $eips; do
            if [ -n "$eip" ]; then
                echo "Releasing Elastic IP: $eip"
                run_command "aws ec2 release-address --allocation-id $eip --region $AWS_REGION" \
                    "Failed to release Elastic IP" \
                    "Successfully released Elastic IP" \
                    "true"
            fi
        done
    done
    
    # 5. Delete VPC Endpoints
    echo "Checking VPC Endpoints..."
    endpoints=$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=$vpc_id" --query "VpcEndpoints[].VpcEndpointId" --output text --region $AWS_REGION)
    
    if [ -n "$endpoints" ]; then
        echo "Deleting VPC Endpoints: $endpoints"
        run_command "aws ec2 delete-vpc-endpoints --vpc-endpoint-ids $endpoints --region $AWS_REGION" \
            "Failed to delete VPC endpoints" \
            "Successfully deleted VPC endpoints" \
            "true"
        
        wait_with_message 15 "Waiting for VPC endpoints to be deleted..."
    fi
    
    # 6. Delete subnets
    echo "Checking subnets..."
    subnets=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc_id" --query "Subnets[].SubnetId" --output text --region $AWS_REGION)
    
    if [ -n "$subnets" ]; then
        echo "Deleting subnets: $subnets"
        for subnet_id in $subnets; do
            run_command "aws ec2 delete-subnet --subnet-id $subnet_id --region $AWS_REGION" \
                "Failed to delete subnet: $subnet_id" \
                "Successfully deleted subnet: $subnet_id" \
                "true"
        done
    fi
    
    # 7. Delete route tables (except the main one)
    echo "Checking route tables..."
    route_tables=$(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$vpc_id" --region $AWS_REGION --query 'RouteTables[?Associations[?Main!=`true`] || !Associations].RouteTableId' --output text)
    
    if [ -n "$route_tables" ]; then
        for rt_id in $route_tables; do
            # First disassociate any associations
            associations=$(aws ec2 describe-route-tables --route-table-ids $rt_id --query 'RouteTables[0].Associations[?!Main].RouteTableAssociationId' --output text --region $AWS_REGION)
            
            for assoc_id in $associations; do
                echo "Disassociating route table association: $assoc_id"
                run_command "aws ec2 disassociate-route-table --association-id $assoc_id --region $AWS_REGION" \
                    "Failed to disassociate route table" \
                    "Successfully disassociated route table" \
                    "true"
            done
            
            echo "Deleting Route Table: $rt_id"
            run_command "aws ec2 delete-route-table --route-table-id $rt_id --region $AWS_REGION" \
                "Failed to delete route table" \
                "Successfully deleted route table" \
                "true"
        done
    fi
    
    # 8. Detach and delete internet gateway
    echo "Checking Internet Gateway..."
    igw=$(aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=$vpc_id" --query 'InternetGateways[0].InternetGatewayId' --output text --region $AWS_REGION)
    
    if [ -n "$igw" ] && [ "$igw" != "None" ]; then
        echo "Detaching Internet Gateway: $igw"
        run_command "aws ec2 detach-internet-gateway --internet-gateway-id $igw --vpc-id $vpc_id --region $AWS_REGION" \
            "Failed to detach internet gateway" \
            "Successfully detached internet gateway" \
            "true"
        
        echo "Deleting Internet Gateway: $igw"
        run_command "aws ec2 delete-internet-gateway --internet-gateway-id $igw --region $AWS_REGION" \
            "Failed to delete internet gateway" \
            "Successfully deleted internet gateway" \
            "true"
    fi
    
    # 9. Delete network ACLs (except default)
    echo "Checking Network ACLs..."
    network_acls=$(aws ec2 describe-network-acls --filters "Name=vpc-id,Values=$vpc_id" --query 'NetworkAcls[?!IsDefault].NetworkAclId' --output text --region $AWS_REGION)
    
    if [ -n "$network_acls" ]; then
        for acl_id in $network_acls; do
            echo "Deleting Network ACL: $acl_id"
            run_command "aws ec2 delete-network-acl --network-acl-id $acl_id --region $AWS_REGION" \
                "Failed to delete Network ACL" \
                "Successfully deleted Network ACL" \
                "true"
        done
    fi
    
    # 10. Wait for a moment before attempting to delete the VPC
    wait_with_message 15 "Final wait before attempting VPC deletion..."
    
    # 11. Delete the VPC with retries
    for attempt in $(seq 1 $max_attempts); do
        echo "Attempt $attempt to delete VPC: $vpc_id"
        
        if aws ec2 delete-vpc --vpc-id $vpc_id --region $AWS_REGION 2>/dev/null; then
            echo -e "${GREEN}Successfully deleted VPC: $vpc_id${NC}"
            return 0
        else
            echo -e "${YELLOW}Failed to delete VPC on attempt $attempt. Checking for remaining dependencies...${NC}"
            
            # Check for remaining ENIs
            remaining_enis=$(aws ec2 describe-network-interfaces --filters "Name=vpc-id,Values=$vpc_id" --query 'NetworkInterfaces[].NetworkInterfaceId' --output text --region $AWS_REGION)
            
            if [ -n "$remaining_enis" ]; then
                echo "Found remaining network interfaces: $remaining_enis"
                has_dependencies=true
                
                for eni in $remaining_enis; do
                    eni_desc=$(aws ec2 describe-network-interfaces --network-interface-ids $eni --region $AWS_REGION --query 'NetworkInterfaces[0].Description' --output text)
                    eni_status=$(aws ec2 describe-network-interfaces --network-interface-ids $eni --region $AWS_REGION --query 'NetworkInterfaces[0].Status' --output text)
                    
                    echo "ENI $eni: $eni_desc (Status: $eni_status)"
                    
                    # Try to force detach if there's an attachment
                    attachment_id=$(aws ec2 describe-network-interfaces --network-interface-ids $eni --region $AWS_REGION --query 'NetworkInterfaces[0].Attachment.AttachmentId' --output text 2>/dev/null || echo "")
                    
                    if [ -n "$attachment_id" ] && [ "$attachment_id" != "None" ]; then
                        echo "Attempting to force-detach: $attachment_id"
                        aws ec2 detach-network-interface --attachment-id $attachment_id --force --region $AWS_REGION 2>/dev/null || true
                        sleep 5
                    fi
                    
                    # Try to delete the ENI
                    echo "Attempting to delete ENI: $eni"
                    aws ec2 delete-network-interface --network-interface-id $eni --region $AWS_REGION 2>/dev/null || true
                done
            fi
            
            # Check for remaining security groups
            remaining_sgs=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$vpc_id" 'Name=group-name,Values=!default' --query "SecurityGroups[].GroupId" --output text --region $AWS_REGION)
            
            if [ -n "$remaining_sgs" ]; then
                echo "Found remaining security groups: $remaining_sgs"
                has_dependencies=true
                
                for sg_id in $remaining_sgs; do
                    # Clear all rules first
                    ingress_rules=$(aws ec2 describe-security-groups --group-ids $sg_id --query 'SecurityGroups[0].IpPermissions' --output json --region $AWS_REGION 2>/dev/null || echo "[]")
                    
                    if [ "$ingress_rules" != "[]" ] && [ "$ingress_rules" != "null" ]; then
                        aws ec2 revoke-security-group-ingress --group-id $sg_id --ip-permissions "$ingress_rules" --region $AWS_REGION 2>/dev/null || true
                    fi
                    
                    # Try to delete the security group
                    echo "Attempting to delete security group: $sg_id"
                    aws ec2 delete-security-group --group-id $sg_id --region $AWS_REGION 2>/dev/null || true
                done
            fi
        fi
    done
    
    # If we reach here, we've exhausted all attempts
    if [ "$has_dependencies" = true ]; then
        echo -e "${RED}Failed to delete VPC attempts due to remaining dependencies.${NC}"
    else
        echo -e "${RED}Failed to delete VPC after attempts.${NC}"
    fi
    
    return 1
}

echo -e "\n${YELLOW}Step 1: Remove all API Gateway resources${NC}"
api_name="${PREFIX}-api"
api_id=$(resource_exists "api" "$api_name")

if [ -n "$api_id" ]; then
    echo "Found API Gateway: $api_id"
    
    # First find and delete all custom domain name mappings for this API
    echo "Checking for custom domain name mappings..."
    domains=$(aws apigateway get-domain-names --region $AWS_REGION --query "items[].domainName" --output text)
    
    for domain in $domains; do
        mappings=$(aws apigateway get-base-path-mappings --domain-name $domain --region $AWS_REGION --query "items[?restApiId=='$api_id'].basePath" --output text 2>/dev/null || echo "")
        
        if [ -n "$mappings" ]; then
            echo -e "${YELLOW}Found domain mapping for API: $domain${NC}"
            for mapping in $mappings; do
                # Handle (none) base path special case
                if [ "$mapping" = "(none)" ]; then
                    mapping=""
                fi
                
                run_command "aws apigateway delete-base-path-mapping --domain-name $domain --base-path \"$mapping\" --region $AWS_REGION" \
                    "Failed to delete base path mapping for domain: $domain, path: $mapping" \
                    "Successfully deleted base path mapping for domain: $domain, path: $mapping" \
                    "true"
            done
        fi
    done
    
    # Then delete the API itself
    run_command "aws apigateway delete-rest-api --rest-api-id $api_id --region $AWS_REGION" \
        "Failed to delete API Gateway: $api_id" \
        "Successfully deleted API Gateway: $api_id"
else
    echo "No API Gateway found with name: $api_name"
fi

echo -e "\n${YELLOW}Step 2: Clear Cognito User Pool${NC}"
user_pool_name="${PREFIX}-user-pool"
user_pool_id=$(resource_exists "cognito" "$user_pool_name")

if [ -n "$user_pool_id" ]; then
    echo "Found Cognito User Pool: $user_pool_id"
    
    # 1. Delete domain first (if exists)
    domain="${PROJECT_NAME}-${STAGE}-auth"
    domain_exists=$(aws cognito-idp describe-user-pool-domain --domain $domain --region $AWS_REGION --query "DomainDescription.Domain" --output text 2>/dev/null || echo "")
    
    if [ -n "$domain_exists" ] && [ "$domain_exists" != "None" ]; then
        echo "Deleting Cognito Domain: $domain"
        run_command "aws cognito-idp delete-user-pool-domain --domain $domain --user-pool-id $user_pool_id --region $AWS_REGION" \
            "Failed to delete Cognito domain: $domain" \
            "Successfully deleted Cognito domain: $domain" \
            "true"
    fi
    
    # 2. Delete app clients
    echo "Finding and deleting User Pool App Clients..."
    client_ids=$(aws cognito-idp list-user-pool-clients --user-pool-id $user_pool_id --region $AWS_REGION --query "UserPoolClients[].ClientId" --output text)
    
    for client_id in $client_ids; do
        if [ -n "$client_id" ]; then
            echo "Deleting App Client: $client_id"
            run_command "aws cognito-idp delete-user-pool-client --user-pool-id $user_pool_id --client-id $client_id --region $AWS_REGION" \
                "Failed to delete Cognito app client: $client_id" \
                "Successfully deleted Cognito app client: $client_id" \
                "true"
        fi
    done
    
    # 3. Delete identity providers if any
    echo "Finding and deleting Identity Providers..."
    providers=$(aws cognito-idp list-identity-providers --user-pool-id $user_pool_id --region $AWS_REGION --query "Providers[].ProviderName" --output text 2>/dev/null || echo "")
    
    for provider in $providers; do
        if [ -n "$provider" ] && [ "$provider" != "COGNITO" ]; then
            echo "Deleting Identity Provider: $provider"
            run_command "aws cognito-idp delete-identity-provider --user-pool-id $user_pool_id --provider-name $provider --region $AWS_REGION" \
                "Failed to delete identity provider: $provider" \
                "Successfully deleted identity provider: $provider" \
                "true"
        fi
    done
    
    # 4. Delete resource servers if any
    echo "Finding and deleting Resource Servers..."
    resource_servers=$(aws cognito-idp list-resource-servers --user-pool-id $user_pool_id --region $AWS_REGION --query "ResourceServers[].Identifier" --output text 2>/dev/null || echo "")
    
    for resource_server in $resource_servers; do
        if [ -n "$resource_server" ]; then
            echo "Deleting Resource Server: $resource_server"
            run_command "aws cognito-idp delete-resource-server --user-pool-id $user_pool_id --identifier $resource_server --region $AWS_REGION" \
                "Failed to delete resource server: $resource_server" \
                "Successfully deleted resource server: $resource_server" \
                "true"
        fi
    done
    
    # 5. Delete user pool
    echo "Deleting User Pool: $user_pool_id"
    run_command "aws cognito-idp delete-user-pool --user-pool-id $user_pool_id --region $AWS_REGION" \
        "Failed to delete Cognito User Pool: $user_pool_id" \
        "Successfully deleted Cognito User Pool: $user_pool_id"
else
    echo "No Cognito User Pool found with name: $user_pool_name"
fi

echo -e "\n${YELLOW}Step 3: Remove all Lambda functions and layers${NC}"
# List of Lambda functions to delete
LAMBDA_FUNCTIONS=(
    "${PREFIX}-auth-handler"
    "${PREFIX}-document-processor"
    "${PREFIX}-query-processor"
    "${PREFIX}-upload-handler"
    "${PREFIX}-db-init"
)

for func in "${LAMBDA_FUNCTIONS[@]}"; do
    # Check if function exists before trying to delete it
    function_name=$(aws lambda list-functions --region $AWS_REGION --query "Functions[?starts_with(FunctionName, '$func')].FunctionName" --output text 2>/dev/null || echo "")
    
    if [ -n "$function_name" ]; then
        echo "Found Lambda function: $function_name"
        
        # 1. First remove event source mappings
        echo "Checking for event source mappings..."
        mappings=$(aws lambda list-event-source-mappings --function-name $function_name --region $AWS_REGION --query "EventSourceMappings[].UUID" --output text 2>/dev/null || echo "")
        
        for mapping in $mappings; do
            if [ -n "$mapping" ]; then
                echo "Deleting event source mapping: $mapping"
                run_command "aws lambda delete-event-source-mapping --uuid $mapping --region $AWS_REGION" \
                    "Failed to delete event source mapping: $mapping" \
                    "Successfully deleted event source mapping: $mapping" \
                    "true"
            fi
        done
        
        # 2. Remove function-url configurations if any
        echo "Checking for function URL configurations..."
        has_url=$(aws lambda get-function-url-config --function-name $function_name --region $AWS_REGION 2>/dev/null && echo "true" || echo "false")
        
        if [ "$has_url" = "true" ]; then
            echo "Deleting function URL configuration for $function_name"
            run_command "aws lambda delete-function-url-config --function-name $function_name --region $AWS_REGION" \
                "Failed to delete function URL for: $function_name" \
                "Successfully deleted function URL for: $function_name" \
                "true"
        fi
        
        # 3. Remove function concurrency if any
        echo "Checking for function concurrency settings..."
        concurrency=$(aws lambda get-function-concurrency --function-name $function_name --region $AWS_REGION --query "ReservedConcurrentExecutions" --output text 2>/dev/null || echo "")
        
        if [ -n "$concurrency" ] && [ "$concurrency" != "None" ]; then
            echo "Removing concurrency settings for $function_name"
            run_command "aws lambda delete-function-concurrency --function-name $function_name --region $AWS_REGION" \
                "Failed to delete function concurrency for: $function_name" \
                "Successfully deleted function concurrency for: $function_name" \
                "true"
        fi
        
        # 4. Delete the function
        echo "Deleting Lambda function: $function_name"
        run_command "aws lambda delete-function --function-name $function_name --region $AWS_REGION" \
            "Failed to delete Lambda function: $function_name" \
            "Successfully deleted Lambda function: $function_name"
    else
        echo "Lambda function not found: $func"
    fi
done

# Find and delete any Lambda layers with the project prefix
echo "Finding Lambda layers with prefix: ${PREFIX}"
layer_versions=$(aws lambda list-layers --region $AWS_REGION --query "Layers[?starts_with(LayerName, '${PREFIX}')]" --output json 2>/dev/null)

if [ -n "$layer_versions" ] && [ "$layer_versions" != "[]" ]; then
    if [ $USE_JQ = true ]; then
        # Process with jq if available
        layer_names=$(echo $layer_versions | jq -r '.[].LayerName')
        for layer_name in $layer_names; do
            echo "Found Lambda layer: $layer_name, retrieving versions..."
            versions=$(aws lambda list-layer-versions --layer-name $layer_name --region $AWS_REGION --query "LayerVersions[].Version" --output text)
            
            for version in $versions; do
                echo "Deleting Lambda layer: $layer_name version $version"
                run_command "aws lambda delete-layer-version --layer-name $layer_name --version-number $version --region $AWS_REGION" \
                    "Failed to delete Lambda layer: $layer_name version $version" \
                    "Successfully deleted Lambda layer: $layer_name version $version" \
                    "true"
            done
        done
    else
        # Simple grep method if jq not available
        echo "Found Lambda layers with prefix ${PREFIX}, but jq is not available for processing."
        echo "Please check and delete Lambda layers manually if needed."
    fi
else
    echo "No Lambda layers found with prefix: ${PREFIX}"
fi

echo -e "\n${YELLOW}Step 4: Delete CloudWatch Log Groups${NC}"
# Delete log groups for Lambda functions
for func in "${LAMBDA_FUNCTIONS[@]}"; do
    log_group="/aws/lambda/$func"
    log_group_exists=$(aws logs describe-log-groups --log-group-name-prefix "$log_group" --region $AWS_REGION --query "logGroups[0].logGroupName" --output text 2>/dev/null || echo "")
    
    if [ -n "$log_group_exists" ] && [ "$log_group_exists" != "None" ]; then
        echo "Found Log Group: $log_group_exists"
        run_command "aws logs delete-log-group --log-group-name \"$log_group_exists\" --region $AWS_REGION" \
            "Failed to delete Log Group: $log_group_exists" \
            "Successfully deleted Log Group: $log_group_exists"
    else
        echo "Log Group not found: $log_group"
    fi
done

# Delete API Gateway log group
api_log_group="/aws/apigateway/${PREFIX}-api"
log_group_exists=$(aws logs describe-log-groups --log-group-name-prefix "$api_log_group" --region $AWS_REGION --query "logGroups[0].logGroupName" --output text 2>/dev/null || echo "")

if [ -n "$log_group_exists" ] && [ "$log_group_exists" != "None" ]; then
    echo "Found API Gateway Log Group: $log_group_exists"
    run_command "aws logs delete-log-group --log-group-name \"$log_group_exists\" --region $AWS_REGION" \
        "Failed to delete API Gateway Log Group: $log_group_exists" \
        "Successfully deleted API Gateway Log Group: $log_group_exists"
else
    echo "API Gateway Log Group not found: $api_log_group"
fi

# Delete VPC flow logs group if it exists
vpc_flow_log_group="/aws/vpc/flowlogs/${PREFIX}"
log_group_exists=$(aws logs describe-log-groups --log-group-name-prefix "$vpc_flow_log_group" --region $AWS_REGION --query "logGroups[0].logGroupName" --output text 2>/dev/null || echo "")

if [ -n "$log_group_exists" ] && [ "$log_group_exists" != "None" ]; then
    echo "Found VPC Flow Log Group: $log_group_exists"
    run_command "aws logs delete-log-group --log-group-name \"$log_group_exists\" --region $AWS_REGION" \
        "Failed to delete VPC Flow Log Group: $log_group_exists" \
        "Successfully deleted VPC Flow Log Group: $log_group_exists"
else
    echo "VPC Flow Log Group not found: $vpc_flow_log_group"
fi

echo -e "\n${YELLOW}Step 5: Delete Secrets Manager secrets${NC}"
SECRETS=(
    "${PREFIX}-db-credentials"
    "${PREFIX}-gemini-api-key"
)

for secret in "${SECRETS[@]}"; do
    # Check if secret exists before trying to delete it
    secret_exists=$(aws secretsmanager list-secrets --region $AWS_REGION --query "SecretList[?Name=='$secret'].Name" --output text 2>/dev/null || echo "")
    
    if [ -n "$secret_exists" ]; then
        echo "Found secret: $secret"
        run_command "aws secretsmanager delete-secret --secret-id \"$secret\" --force-delete-without-recovery --region $AWS_REGION" \
            "Failed to delete secret: $secret" \
            "Successfully deleted secret: $secret"
    else
        echo "Secret not found: $secret"
    fi
done

echo -e "\n${YELLOW}Step 6: Delete SNS topics${NC}"
# Find and delete SNS topics with the project name
topics=$(aws sns list-topics --region $AWS_REGION --query "Topics[?contains(TopicArn, '${PREFIX}')].TopicArn" --output text)

if [ -n "$topics" ]; then
    for topic_arn in $topics; do
        echo "Found SNS topic: $topic_arn"
        
        # Delete all subscriptions first
        echo "Finding and deleting topic subscriptions..."
        subscriptions=$(aws sns list-subscriptions-by-topic --topic-arn $topic_arn --region $AWS_REGION --query "Subscriptions[].SubscriptionArn" --output text)
        
        for sub_arn in $subscriptions; do
            if [ "$sub_arn" != "PendingConfirmation" ]; then
                echo "Deleting subscription: $sub_arn"
                run_command "aws sns unsubscribe --subscription-arn $sub_arn --region $AWS_REGION" \
                    "Failed to delete subscription: $sub_arn" \
                    "Successfully deleted subscription: $sub_arn" \
                    "true"
            fi
        done
        
        # Delete the topic itself
        run_command "aws sns delete-topic --topic-arn $topic_arn --region $AWS_REGION" \
            "Failed to delete SNS topic: $topic_arn" \
            "Successfully deleted SNS topic: $topic_arn"
    done
else
    echo "No SNS topics found with name containing: ${PREFIX}"
fi

echo -e "\n${YELLOW}Step 7: Delete CloudWatch Alarms${NC}"
# Find and delete CloudWatch alarms with the project name
alarms=$(aws cloudwatch describe-alarms --region $AWS_REGION --query "MetricAlarms[?contains(AlarmName, '${PREFIX}')].AlarmName" --output text)

if [ -n "$alarms" ]; then
    for alarm in $alarms; do
        echo "Found CloudWatch alarm: $alarm"
        run_command "aws cloudwatch delete-alarms --alarm-names \"$alarm\" --region $AWS_REGION" \
            "Failed to delete CloudWatch alarm: $alarm" \
            "Successfully deleted CloudWatch alarm: $alarm"
    done
else
    echo "No CloudWatch alarms found with name containing: ${PREFIX}"
fi

echo -e "\n${YELLOW}Step 8: Delete RDS instances and related resources${NC}"
DB_INSTANCE="${PREFIX}-postgres"
DB_PARAM_GROUP="${PREFIX}-postgres-params"
DB_SUBNET_GROUP="${PREFIX}-db-subnet-group"

# Check if RDS instance exists
db_exists=$(aws rds describe-db-instances --region $AWS_REGION --query "DBInstances[?DBInstanceIdentifier=='$DB_INSTANCE'].DBInstanceIdentifier" --output text 2>/dev/null || echo "")

if [ -n "$db_exists" ]; then
    echo "Found RDS instance: $db_exists"
    
    # 1. Disable deletion protection first if enabled
    protection=$(aws rds describe-db-instances --db-instance-identifier $db_exists --region $AWS_REGION --query "DBInstances[0].DeletionProtection" --output text)
    
    if [ "$protection" = "true" ]; then
        echo "RDS instance has deletion protection enabled. Disabling..."
        run_command "aws rds modify-db-instance --db-instance-identifier $db_exists --no-deletion-protection --apply-immediately --region $AWS_REGION" \
            "Failed to disable deletion protection for RDS instance: $db_exists" \
            "Successfully disabled deletion protection for RDS instance: $db_exists"
        
        # Wait for the modification to complete
        echo "Waiting for modification to complete..."
        run_command "aws rds wait db-instance-available --db-instance-identifier $db_exists --region $AWS_REGION" \
            "Failed to wait for RDS instance modification" \
            "RDS instance modification completed"
    fi
    
    # 2. Delete DB instance, skipping final snapshot and deleting automated backups
    echo "Deleting RDS instance: $db_exists"
    run_command "aws rds delete-db-instance --db-instance-identifier $db_exists --skip-final-snapshot --delete-automated-backups --region $AWS_REGION" \
        "Failed to delete RDS instance: $db_exists" \
        "Successfully initiated deletion of RDS instance: $db_exists"
    
    # Wait for DB instance to be deleted before continuing
    echo "Waiting for RDS instance deletion to complete (this may take several minutes)..."
    wait_with_message 60 "Waiting for RDS instance deletion (checking every 60 seconds)..."
    
    # We'll use a loop to check status with increasing wait times
    for attempt in {1..5}; do
        status=$(aws rds describe-db-instances --db-instance-identifier $db_exists --region $AWS_REGION --query "DBInstances[0].DBInstanceStatus" --output text 2>/dev/null || echo "deleted")
        
        if [ "$status" = "deleted" ] || [ -z "$status" ]; then
            echo -e "${GREEN}RDS instance deletion completed.${NC}"
            break
        else
            echo "RDS instance status: $status. Waiting longer..."
            wait_time=$((30 + $attempt * 30))  # Increasing wait time with each attempt
            wait_with_message $wait_time "Still waiting for RDS deletion to complete..."
        fi
    done
else
    echo "RDS instance not found: $DB_INSTANCE"
fi

# Delete parameter group (regardless of whether instance was found)
param_group_exists=$(aws rds describe-db-parameter-groups --region $AWS_REGION --query "DBParameterGroups[?DBParameterGroupName=='$DB_PARAM_GROUP'].DBParameterGroupName" --output text 2>/dev/null || echo "")

if [ -n "$param_group_exists" ]; then
    echo "Found DB parameter group: $param_group_exists"
    run_command "aws rds delete-db-parameter-group --db-parameter-group-name $param_group_exists --region $AWS_REGION" \
        "Failed to delete DB parameter group: $param_group_exists" \
        "Successfully deleted DB parameter group: $param_group_exists" \
        "true"
else
    echo "DB parameter group not found: $DB_PARAM_GROUP"
fi

# Delete subnet group (regardless of whether instance was found)
subnet_group_exists=$(aws rds describe-db-subnet-groups --region $AWS_REGION --query "DBSubnetGroups[?DBSubnetGroupName=='$DB_SUBNET_GROUP'].DBSubnetGroupName" --output text 2>/dev/null || echo "")

if [ -n "$subnet_group_exists" ]; then
    echo "Found DB subnet group: $subnet_group_exists"
    run_command "aws rds delete-db-subnet-group --db-subnet-group-name $subnet_group_exists --region $AWS_REGION" \
        "Failed to delete DB subnet group: $subnet_group_exists" \
        "Successfully deleted DB subnet group: $subnet_group_exists" \
        "true"
else
    echo "DB subnet group not found: $DB_SUBNET_GROUP"
fi


echo -e "\n${YELLOW}Step 9: Delete DynamoDB tables${NC}"
TABLES=(
    "${PREFIX}-metadata"
    "${PREFIX}-terraform-state-lock"
)

for table in "${TABLES[@]}"; do
    table_exists=$(aws dynamodb list-tables --region $AWS_REGION --query "TableNames[?contains(@, '$table')]" --output text 2>/dev/null || echo "")
    
    if [ -n "$table_exists" ]; then
        echo "Found DynamoDB table: $table_exists"
        run_command "aws dynamodb delete-table --table-name $table_exists --region $AWS_REGION" \
            "Failed to delete DynamoDB table: $table_exists" \
            "Successfully deleted DynamoDB table: $table_exists"
    else
        echo "DynamoDB table not found: $table"
    fi
done

echo -e "\n${YELLOW}Step 10: Empty and delete S3 buckets${NC}"
BUCKETS=(
    "${PREFIX}-documents"
    "${PREFIX}-lambda-code"
    "${PROJECT_NAME}-terraform-state"
)

for bucket in "${BUCKETS[@]}"; do
    # Check if bucket exists
    bucket_exists=$(aws s3api list-buckets --query "Buckets[?Name=='$bucket'].Name" --output text 2>/dev/null || echo "")
    
    if [ -n "$bucket_exists" ]; then
        echo "Found S3 bucket: $bucket_exists"
        
        # Check if bucket versioning is enabled
        versioning=$(aws s3api get-bucket-versioning --bucket $bucket_exists --region $AWS_REGION --query "Status" --output text 2>/dev/null || echo "")
        
        # Disable versioning first if enabled
        if [ "$versioning" = "Enabled" ]; then
            echo "Bucket versioning is enabled. Suspending..."
            run_command "aws s3api put-bucket-versioning --bucket $bucket_exists --versioning-configuration Status=Suspended --region $AWS_REGION" \
                "Failed to suspend bucket versioning for: $bucket_exists" \
                "Successfully suspended versioning for bucket: $bucket_exists" \
                "true"
        fi
        
        # Special handling for terraform state bucket
        if [[ $bucket == *"terraform-state"* ]]; then
            echo "This is a terraform state bucket. Will only remove objects for stage: ${STAGE}"
            echo "Deleting objects under stage prefix: ${STAGE}/"
            run_command "aws s3 rm s3://$bucket_exists/$STAGE/ --recursive --region $AWS_REGION" \
                "Failed to delete terraform state objects" \
                "Successfully deleted terraform state objects" \
                "true"
            
            # Don't delete the terraform state bucket, just move on
            echo "Skipping deletion of terraform state bucket to preserve other environments"
            continue
        fi
        
        echo "Emptying bucket: $bucket_exists"
        
        # Step 1: Remove all standard objects first
        run_command "aws s3 rm s3://$bucket_exists --recursive --region $AWS_REGION" \
            "Failed to delete standard objects from bucket: $bucket_exists" \
            "Deleted standard objects from bucket: $bucket_exists" \
            "true"
        
        # Step 2: Delete all versions and delete markers if versioning was enabled
        if [ "$versioning" = "Enabled" ] || [ "$versioning" = "Suspended" ]; then
            echo "Checking for object versions and delete markers..."
            
            if [ $USE_JQ = true ]; then
                # Use jq to process versions
                versions_json=$(aws s3api list-object-versions --bucket $bucket_exists --region $AWS_REGION --output json 2>/dev/null)
                
                # Process versions
                version_keys=$(echo "$versions_json" | jq -r '.Versions[]? | "\(.Key)|\(.VersionId)"' 2>/dev/null || echo "")
                for version_info in $version_keys; do
                    IFS='|' read -r key version_id <<< "$version_info"
                    if [ -n "$key" ] && [ -n "$version_id" ]; then
                        echo "Deleting version: $key ($version_id)"
                        run_command "aws s3api delete-object --bucket $bucket_exists --key \"$key\" --version-id \"$version_id\" --region $AWS_REGION" \
                            "" "" "true"
                    fi
                done
                
                # Process delete markers
                marker_keys=$(echo "$versions_json" | jq -r '.DeleteMarkers[]? | "\(.Key)|\(.VersionId)"' 2>/dev/null || echo "")
                for marker_info in $marker_keys; do
                    IFS='|' read -r key version_id <<< "$marker_info"
                    if [ -n "$key" ] && [ -n "$version_id" ]; then
                        echo "Deleting delete marker: $key ($version_id)"
                        run_command "aws s3api delete-object --bucket $bucket_exists --key \"$key\" --version-id \"$version_id\" --region $AWS_REGION" \
                            "" "" "true"
                    fi
                done
            else
                # Alternative approach without jq
                echo "Processing without jq - this may not delete all versions"
                aws s3api list-object-versions --bucket $bucket_exists --region $AWS_REGION --output text | \
                grep -E 'VERSIONS|DELETE' | while read -r line; do
                    key=$(echo "$line" | awk '{print $4}')
                    version_id=$(echo "$line" | awk '{print $6}')
                    if [ -n "$key" ] && [ -n "$version_id" ]; then
                        echo "Deleting object: $key ($version_id)"
                        aws s3api delete-object --bucket $bucket_exists --key "$key" --version-id "$version_id" --region $AWS_REGION >/dev/null 2>&1
                    fi
                done
            fi
        fi
        
        # Step 3: Final check to make sure bucket is empty
        echo "Performing final verification that bucket is empty..."
        run_command "aws s3 rm s3://$bucket_exists --recursive --region $AWS_REGION" \
            "Failed during final cleanup check" \
            "Bucket verified empty" \
            "true"
        
        # Step 4: Delete the bucket
        echo "Deleting bucket: $bucket_exists"
        run_command "aws s3api delete-bucket --bucket $bucket_exists --region $AWS_REGION" \
            "Failed to delete bucket: $bucket_exists" \
            "Successfully deleted bucket: $bucket_exists"
    else
        echo "S3 bucket not found: $bucket"
    fi
done

echo -e "\n${YELLOW}Step 11: Delete IAM roles and policies${NC}"
# Find IAM roles with the project prefix
roles=$(aws iam list-roles --query "Roles[?starts_with(RoleName, '${PREFIX}')].RoleName" --output text)

if [ -n "$roles" ]; then
    for role_name in $roles; do
        echo "Found IAM role: $role_name"
        
        # Step 1: Remove all attached policies
        echo "Finding and detaching policies for role: $role_name"
        attached_policies=$(aws iam list-attached-role-policies --role-name $role_name --query "AttachedPolicies[].PolicyArn" --output text)
        
        for policy_arn in $attached_policies; do
            echo "Detaching policy: $policy_arn from role: $role_name"
            run_command "aws iam detach-role-policy --role-name $role_name --policy-arn $policy_arn --region $AWS_REGION" \
                "Failed to detach policy: $policy_arn" \
                "Successfully detached policy: $policy_arn" \
                "true"
        done
        
        # Step 2: Delete all inline policies
        echo "Finding and deleting inline policies for role: $role_name"
        inline_policies=$(aws iam list-role-policies --role-name $role_name --query "PolicyNames[]" --output text)
        
        for policy_name in $inline_policies; do
            echo "Deleting inline policy: $policy_name from role: $role_name"
            run_command "aws iam delete-role-policy --role-name $role_name --policy-name $policy_name" \
                "Failed to delete inline policy: $policy_name" \
                "Successfully deleted inline policy: $policy_name" \
                "true"
        done
        
        # Step 3: Check if the role has any instance profiles
        echo "Checking for instance profiles attached to role: $role_name"
        instance_profiles=$(aws iam list-instance-profiles-for-role --role-name $role_name --query "InstanceProfiles[].InstanceProfileName" --output text)
        
        for profile_name in $instance_profiles; do
            echo "Removing role from instance profile: $profile_name"
            run_command "aws iam remove-role-from-instance-profile --instance-profile-name $profile_name --role-name $role_name" \
                "Failed to remove role from instance profile: $profile_name" \
                "Successfully removed role from instance profile: $profile_name" \
                "true"
        done
        
        # Step 4: Delete the role
        echo "Deleting IAM role: $role_name"
        run_command "aws iam delete-role --role-name $role_name" \
            "Failed to delete IAM role: $role_name" \
            "Successfully deleted IAM role: $role_name"
    done
else
    echo "No IAM roles found with prefix: ${PREFIX}"
fi

# Find and delete IAM policies with the project prefix
echo "Finding IAM policies with prefix: ${PREFIX}"
policies=$(aws iam list-policies --scope Local --query "Policies[?starts_with(PolicyName, '${PREFIX}')].Arn" --output text)

if [ -n "$policies" ]; then
    for policy_arn in $policies; do
        echo "Found IAM policy: $policy_arn"
        
        # Delete policy versions first (except default)
        echo "Finding and deleting non-default versions for policy: $policy_arn"
        policy_versions=$(aws iam list-policy-versions --policy-arn $policy_arn --query "Versions[?!IsDefaultVersion].VersionId" --output text)
        
        for version_id in $policy_versions; do
            echo "Deleting policy version: $version_id"
            run_command "aws iam delete-policy-version --policy-arn $policy_arn --version-id $version_id" \
                "Failed to delete policy version: $version_id" \
                "Successfully deleted policy version: $version_id" \
                "true"
        done
        
        # Delete the policy
        echo "Deleting IAM policy: $policy_arn"
        run_command "aws iam delete-policy --policy-arn $policy_arn" \
            "Failed to delete IAM policy: $policy_arn" \
            "Successfully deleted IAM policy: $policy_arn"
    done
else
    echo "No IAM policies found with prefix: ${PREFIX}"
fi

echo -e "\n${YELLOW}Step 12: Delete all Security Groups${NC}"
# Get all security groups in the VPC except the default one
security_groups=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=${PREFIX}-*" --query "SecurityGroups[].GroupId" --output text --region $AWS_REGION)

if [ -n "$security_groups" ]; then
    echo "Found security groups to delete: $security_groups"
    
    # First remove all rules to break dependencies
    for sg_id in $security_groups; do
        echo "Removing all rules from security group: $sg_id"
        
        # Get all ingress rules
        ingress_rules=$(aws ec2 describe-security-groups --group-ids $sg_id --query "SecurityGroups[0].IpPermissions" --output json --region $AWS_REGION 2>/dev/null || echo "[]")
        
        if [ "$ingress_rules" != "[]" ] && [ "$ingress_rules" != "null" ]; then
            echo "Removing ingress rules..."
            run_command "aws ec2 revoke-security-group-ingress --group-id $sg_id --ip-permissions '$ingress_rules' --region $AWS_REGION" \
                "Failed to remove ingress rules" \
                "Successfully removed ingress rules" \
                "true"
        fi
        
        # Get all egress rules except the default one
        egress_rules=$(aws ec2 describe-security-groups --group-ids $sg_id --query "SecurityGroups[0].IpPermissionsEgress[?!(IpProtocol == '-1' && CidrIp == '0.0.0.0/0')]" --output json --region $AWS_REGION 2>/dev/null || echo "[]")
        
        if [ "$egress_rules" != "[]" ] && [ "$egress_rules" != "null" ]; then
            echo "Removing non-default egress rules..."
            run_command "aws ec2 revoke-security-group-egress --group-id $sg_id --ip-permissions '$egress_rules' --region $AWS_REGION" \
                "Failed to remove egress rules" \
                "Successfully removed egress rules" \
                "true"
        fi
    done
    
    # Wait for rule changes to propagate
    wait_with_message 10 "Waiting for rule changes to propagate..."
    
    # Now delete the security groups
    for sg_id in $security_groups; do
        echo "Deleting security group: $sg_id"
        # Make multiple attempts with increasing wait times
        for attempt in {1..3}; do
            if run_command "aws ec2 delete-security-group --group-id $sg_id --region $AWS_REGION" \
                "Failed to delete security group on attempt $attempt" \
                "Successfully deleted security group: $sg_id" \
                "true"; then
                break
            else
                # Check if the security group still exists
                if ! aws ec2 describe-security-groups --group-ids $sg_id --region $AWS_REGION &>/dev/null; then
                    echo "Security group no longer exists, deletion was successful"
                    break
                fi
                wait_with_message $((5 * attempt)) "Waiting before retry..."
            fi
        done
    done
else
    echo "No non-default security groups found in the VPC."
fi

# Streamlined VPC deletion process
echo -e "\n${YELLOW}Step 13: Streamlined VPC deletion process${NC}"
# Find VPCs matching project
echo "Looking for VPCs associated with project: ${PREFIX}"

# Try multiple methods to find the VPC
vpc_id=""

# Method 1: Look by tag name
vpc_id=$(aws ec2 describe-vpcs --filters "Name=tag:Name,Values=${PREFIX}-vpc" --query "Vpcs[0].VpcId" --output text --region $AWS_REGION 2>/dev/null)

# Method 2: Look for VPCs with matching tag value
if [ -z "$vpc_id" ] || [ "$vpc_id" = "None" ]; then
    vpc_id=$(aws ec2 describe-vpcs --region $AWS_REGION --query "Vpcs[?Tags[?Value=='${PREFIX}-vpc']].VpcId" --output text 2>/dev/null)
fi

# Method 3: Look for VPCs with prefix in tags
if [ -z "$vpc_id" ] || [ "$vpc_id" = "None" ]; then
    vpc_id=$(aws ec2 describe-vpcs --region $AWS_REGION --query "Vpcs[?Tags[?contains(Value, '${PREFIX}')]].VpcId" --output text 2>/dev/null)
fi

# Method 4: Find VPC ID by a security group name matching our prefix
if [ -z "$vpc_id" ] || [ "$vpc_id" = "None" ]; then
    sg_info=$(aws ec2 describe-security-groups --region $AWS_REGION --filters "Name=group-name,Values=${PREFIX}-*" --query "SecurityGroups[0].VpcId" --output text 2>/dev/null)
    if [ -n "$sg_info" ] && [ "$sg_info" != "None" ]; then
        vpc_id=$sg_info
    fi
fi

# Function to delete VPC
if [ -n "$vpc_id" ] && [ "$vpc_id" != "None" ]; then
    echo -e "${GREEN}Found VPC to delete: $vpc_id${NC}"
    delete_vpc "$vpc_id"
else
    echo -e "${YELLOW}No VPC found matching the project prefix.${NC}"
fi

wait_with_message 900 "Final wait for 15 minutes to allow remaining ENIs, VPCs, and associated resources to become eligible for deletion, as some resources take time to detach and become available..."

echo -e "\n${YELLOW}Step 16: Clean up still unattached ENIs${NC}"
echo "Searching for any unattached ENIs across the account..."
unattached_enis=$(aws ec2 describe-network-interfaces --filters "Name=status,Values=available" --query "NetworkInterfaces[].NetworkInterfaceId" --output text --region $AWS_REGION)

if [ -n "$unattached_enis" ]; then
    echo "Found unattached ENIs: $unattached_enis"
    for eni in $unattached_enis; do
        # Check if ENI description contains our project prefix
        eni_desc=$(aws ec2 describe-network-interfaces --network-interface-ids $eni --region $AWS_REGION --query "NetworkInterfaces[0].Description" --output text)
        if [[ $eni_desc == *"${PREFIX}"* ]]; then
            echo "Deleting unattached ENI: $eni (Description: $eni_desc)"
            run_command "aws ec2 delete-network-interface --network-interface-id $eni --region $AWS_REGION" \
                "Failed to delete ENI" \
                "Successfully deleted ENI" \
                "true"
        else
            echo "Skipping ENI $eni as it doesn't appear to be related to our project"
        fi
    done
else
    echo "No unattached ENIs found."
fi

# Now try to delete the VPC and any remaining resources pending oen final time
echo -e "\n${YELLOW}Step 17.5: Attempting final VPC deletion${NC}"
echo "Attempting final VPC and remaining pending resources deletion..."
if [[ "$vpc_id" != "None" && -n "$vpc_id" ]]; then
    echo "Found VPC: $vpc_id. Checking and deleting dependencies..."

    echo "Check and delete if subnet exist..."
    subnets=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc_id" --query "Subnets[].SubnetId" --output text --region "$AWS_REGION")
    for subnet_id in $subnets; do
        echo "Deleting subnet: $subnet_id"
        run_command "aws ec2 delete-subnet --subnet-id $subnet_id --region $AWS_REGION" \
                "Failed to delete subnet $subnet_id" \
                "Successfully deleted subnet $subnet_id" \
                "true"
    done

    echo "Detach and Delete Internet Gateways if exist..."
    igws=$(aws ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=$vpc_id" --query "InternetGateways[].InternetGatewayId" --output text --region "$AWS_REGION")
    for igw in $igws; do
        echo "Detaching and deleting internet gateway: $igw"
        run_command "aws ec2 detach-internet-gateway --internet-gateway-id $igw --vpc-id $vpc_id --region $AWS_REGION" \
                "Failed to detach Internet gateway" \
                "Successfully detached Internet gateway " \
                "true"
        run_command "aws ec2 delete-internet-gateway --internet-gateway-id $igw --region $AWS_REGION" \
                "Failed to delete Internet gateway" \
                "Successfully delete Internet gateway " \
                "true"
    done

    echo "Delete NAT Gateways and release associated Elastic IPs if exist..."
    nat_gws=$(aws ec2 describe-nat-gateways --filter Name=vpc-id,Values=$vpc_id --query "NatGateways[].NatGatewayId" --output text --region "$AWS_REGION")
    for natgw in $nat_gws; do
        echo "Deleting NAT Gateway: $natgw"
        # Get EIP Allocation ID
        eip_alloc_ids=$(aws ec2 describe-nat-gateways --nat-gateway-ids "$natgw" --query "NatGateways[].NatGatewayAddresses[].AllocationId" --output text --region "$AWS_REGION")
        run_command "aws ec2 delete-nat-gateway --nat-gateway-id $natgw --region $AWS_REGION" \
            "Failed to delete NAT gateway" \
            "Successfully delete NAT gateway " \
            "true"
        # NAT Gateway deletion takes time - wait is advised in production
        for alloc_id in $eip_alloc_ids; do
            echo "Releasing Elastic IP Allocation ID: $alloc_id"
            run_command "aws ec2 release-address --allocation-id $alloc_id --region $AWS_REGION" \
                "Failed to delete NAT gateway" \
                "Successfully delete NAT gateway" \
                "true"
        done
    done

    echo "Delete Route Tables (except main) if exist..."
    route_tables=$(aws ec2 describe-route-tables --filters "Name=vpc-id,Values=$vpc_id" --query "RouteTables[?Associations[?Main==\`false\`]].RouteTableId" --output text --region "$AWS_REGION")
    for rt in $route_tables; do
        echo "Deleting route table: $rt"
        run_command "aws ec2 delete-route-table --route-table-id $rt --region $AWS_REGION" \
            "Failed to delete route table $rt" \
            "Successfully deleted route table $rt" \
            "true"
    done

    echo "Release Elastic IPs matching name prefix if exist..."
    eips_to_release=$(aws ec2 describe-addresses --region "$AWS_REGION" \
        --query "Addresses[?Tags[?Key=='Name' && starts_with(Value, '${PREFIX}-')] && Domain=='vpc'].AllocationId" \
        --output text)

    for alloc_id in $eips_to_release; do
        echo "Releasing Elastic IP with Allocation ID: $alloc_id"
        run_command "aws ec2 release-address --allocation-id $alloc_id --region $AWS_REGION" \
            "Failed to release EIP $alloc_id" \
            "Successfully released EIP $alloc_id" \
            "true"
    done

    echo "Delete Security Groups if exist..."
    # Get all security groups in the VPC except the default one
    security_groups=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=${PREFIX}-*" --query "SecurityGroups[].GroupId" --output text --region $AWS_REGION)

    if [ -n "$security_groups" ]; then
        echo "Found security groups to delete: $security_groups"
        
        # First remove all rules to break dependencies
        for sg_id in $security_groups; do
            echo "Removing all rules from security group: $sg_id"
            
            # Get all ingress rules
            ingress_rules=$(aws ec2 describe-security-groups --group-ids $sg_id --query "SecurityGroups[0].IpPermissions" --output json --region $AWS_REGION 2>/dev/null || echo "[]")
            
            if [ "$ingress_rules" != "[]" ] && [ "$ingress_rules" != "null" ]; then
                echo "Removing ingress rules..."
                run_command "aws ec2 revoke-security-group-ingress --group-id $sg_id --ip-permissions '$ingress_rules' --region $AWS_REGION" \
                    "Failed to remove ingress rules" \
                    "Successfully removed ingress rules" \
                    "true"
            fi
            
            # Get all egress rules except the default one
            egress_rules=$(aws ec2 describe-security-groups --group-ids $sg_id --query "SecurityGroups[0].IpPermissionsEgress[?!(IpProtocol == '-1' && CidrIp == '0.0.0.0/0')]" --output json --region $AWS_REGION 2>/dev/null || echo "[]")
            
            if [ "$egress_rules" != "[]" ] && [ "$egress_rules" != "null" ]; then
                echo "Removing non-default egress rules..."
                run_command "aws ec2 revoke-security-group-egress --group-id $sg_id --ip-permissions '$egress_rules' --region $AWS_REGION" \
                    "Failed to remove egress rules" \
                    "Successfully removed egress rules" \
                    "true"
            fi
        done
        
        # Wait for rule changes to propagate
        wait_with_message 10 "Waiting for rule changes to propagate..."
        
        # Now delete the security groups
        for sg_id in $security_groups; do
            echo "Deleting security group: $sg_id"
            # Make multiple attempts with increasing wait times
            for attempt in {1..3}; do
                if run_command "aws ec2 delete-security-group --group-id $sg_id --region $AWS_REGION" \
                    "Failed to delete security group on attempt $attempt" \
                    "Successfully deleted security group: $sg_id" \
                    "true"; then
                    break
                else
                    # Check if the security group still exists
                    if ! aws ec2 describe-security-groups --group-ids $sg_id --region $AWS_REGION &>/dev/null; then
                        echo "Security group no longer exists, deletion was successful"
                        break
                    fi
                    wait_with_message $((5 * attempt)) "Waiting before retry..."
                fi
            done
        done
    else
        echo "No non-default security groups found in the VPC."
    fi

    echo "Final VPC Deletion..."
    aws ec2 delete-vpc --vpc-id "$vpc_id" --region "$AWS_REGION" \
        && echo -e "${GREEN}VPC deleted: $vpc_id${NC}" \
        || echo -e "${YELLOW}VPC deletion failed - remaining dependencies likely exist.${NC}"
else
    echo "No VPC found or already deleted."
fi

echo -e "\n${YELLOW}Step 18: Final verification${NC}"
echo "Performing final verification of resource cleanup..."

# Define checks for different resource types with resource count output
check_remaining_resources() {
    local resource_type=$1
    local check_command=$2
    local resource_count=0
    
    echo -e "${BLUE}Checking for remaining $resource_type...${NC}"
    if [ -n "$check_command" ]; then
        resource_list=$(eval "$check_command")
        if [ -n "$resource_list" ]; then
            resource_count=$(echo "$resource_list" | wc -w)
            echo -e "${YELLOW}Found $resource_count $resource_type still existing:${NC}"
            echo "$resource_list"
            return 1
        else
            echo -e "${GREEN}No $resource_type found.${NC}"
            return 0
        fi
    fi
}

# Define checks for different resource types
lambda_check="aws lambda list-functions --region $AWS_REGION --query \"Functions[?contains(FunctionName, '${PREFIX}')].FunctionName\" --output text"
api_check="aws apigateway get-rest-apis --region $AWS_REGION --query \"items[?contains(name, '${PREFIX}')].id\" --output text"
dynamo_check="aws dynamodb list-tables --region $AWS_REGION --query \"TableNames[?contains(@, '${PREFIX}')]\" --output text"
s3_check="aws s3api list-buckets --query \"Buckets[?contains(Name, '${PREFIX}')].Name\" --output text"
rds_check="aws rds describe-db-instances --region $AWS_REGION --query \"DBInstances[?contains(DBInstanceIdentifier, '${PREFIX}')].DBInstanceIdentifier\" --output text"
cognito_check="aws cognito-idp list-user-pools --max-results 20 --region $AWS_REGION --query \"UserPools[?contains(Name, '${PREFIX}')].Id\" --output text"
secret_check="aws secretsmanager list-secrets --region $AWS_REGION --query \"SecretList[?contains(Name, '${PREFIX}')].Name\" --output text"
logs_check="aws logs describe-log-groups --region $AWS_REGION --query \"logGroups[?contains(logGroupName, '${PREFIX}')].logGroupName\" --output text"
role_check="aws iam list-roles --query \"Roles[?contains(RoleName, '${PREFIX}')].RoleName\" --output text"
policy_check="aws iam list-policies --scope Local --query \"Policies[?contains(PolicyName, '${PREFIX}')].PolicyName\" --output text"

# Create an array of resource types and their check commands
declare -A resource_checks
resource_checks["Lambda functions"]="$lambda_check"
resource_checks["API Gateway APIs"]="$api_check"
resource_checks["DynamoDB tables"]="$dynamo_check"
resource_checks["S3 buckets"]="$s3_check"
resource_checks["RDS instances"]="$rds_check"
resource_checks["Cognito User Pools"]="$cognito_check"
resource_checks["Secrets Manager secrets"]="$secret_check"
resource_checks["CloudWatch Log Groups"]="$logs_check"
resource_checks["IAM roles"]="$role_check"
resource_checks["IAM policies"]="$policy_check"

# Check all resource types
failed_checks=0
for resource_type in "${!resource_checks[@]}"; do
    if ! check_remaining_resources "$resource_type" "${resource_checks[$resource_type]}"; then
        failed_checks=$((failed_checks + 1))
    fi
done

# Final cleanup status
echo -e "\n${BLUE}============================================================${NC}"
if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}All resources have been successfully cleaned up!${NC}"
else
    echo -e "${YELLOW}Cleanup completed with $failed_checks resource types still remaining.${NC}"
    echo -e "${YELLOW}Some resources may require manual deletion through the AWS Console.${NC}"
fi
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${GREEN}Summary of cleanup actions:${NC}"
echo "- Removed API Gateway resources"
echo "- Deleted Cognito User Pool and related resources"
echo "- Removed Lambda functions and layers"
echo "- Deleted CloudWatch Log Groups"
echo "- Removed Secrets Manager secrets"
echo "- Deleted SNS topics and subscriptions"
echo "- Removed CloudWatch Alarms"
echo "- Deleted RDS instances and related resources"
echo "- Removed DynamoDB tables"
echo "- Emptied and deleted S3 buckets"
echo "- Removed IAM roles and policies"
echo "- Deleted Security Groups"
echo "- Deleted VPC and all associated resources"

echo -e "\n${YELLOW}Note: The Terraform state bucket '${PROJECT_NAME}-terraform-state' was not fully deleted to preserve${NC}"
echo -e "${YELLOW}state files for other environments. Only the ${STAGE}/ folder was cleaned up.${NC}"
echo -e "\n${BLUE}Cleanup process completed.${NC}"