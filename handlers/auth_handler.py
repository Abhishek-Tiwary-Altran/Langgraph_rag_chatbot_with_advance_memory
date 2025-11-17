import boto3
from typing import Dict

class AuthHandler:
    def __init__(self, credentials):
        self.cognito_client = boto3.client(
            'cognito-idp',
            aws_access_key_id=credentials['aws_access_key_id'],
            aws_secret_access_key=credentials['aws_secret_access_key'],
            region_name=credentials['region_name']
        )
        self.user_pool_id = credentials['cognito']['user_pool_id']
        self.client_id = credentials['cognito']['client_id']

    def sign_up(self, username: str, password: str, email: str) -> Dict:
        try:
            response = self.cognito_client.sign_up(
                ClientId=self.client_id,
                Username=username,
                Password=password,
                UserAttributes=[
                    {'Name': 'email', 'Value': email}
                ]
            )
            return {'success': True, 'message': 'User registered successfully'}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def confirm_sign_up(self, username: str, confirmation_code: str) -> Dict:
        try:
            self.cognito_client.confirm_sign_up(
                ClientId=self.client_id,
                Username=username,
                ConfirmationCode=confirmation_code
            )
            return {'success': True, 'message': 'Email confirmed successfully'}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def sign_in(self, username, password):
        try:
            auth_response = self.cognito_client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': username,
                    'PASSWORD': password
                }
            )
            
            auth_result = auth_response['AuthenticationResult']
            return {
                'success': True,
                'token': auth_result['IdToken'],
                'access_token': auth_result['AccessToken'],
                'refresh_token': auth_result['RefreshToken']
            }
        except self.cognito_client.exceptions.NotAuthorizedException:
            return {'success': False, 'message': 'Invalid username or password'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
        
    def refresh_token(self, refresh_token):
        try:
            response = self.cognito_client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow='REFRESH_TOKEN_AUTH',
                AuthParameters={
                    'REFRESH_TOKEN': refresh_token
                }
            )
            return {
                'success': True,
                'token': response['AuthenticationResult']['IdToken'],
                'refresh_token': refresh_token
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}
