import streamlit as st

def login_page(auth_handler):
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Confirm Sign Up"])
        
        with tab1:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                result = auth_handler.sign_in(username, password)
                
                if result['success']:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.id_token = result['token']
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(result['message'])

        with tab2:
            new_username = st.text_input("Username", key="signup_username")
            new_password = st.text_input("Password", type="password", key="signup_password")
            email = st.text_input("Email")
            
            if st.button("Sign Up"):
                result = auth_handler.sign_up(new_username, new_password, email)
                
                if result['success']:
                    st.success("Please check your email for confirmation code")
                else:
                    st.error(result['message'])

        with tab3:
            confirm_username = st.text_input("Username", key="confirm_username")
            confirmation_code = st.text_input("Confirmation Code")
            
            if st.button("Confirm"):
                result = auth_handler.confirm_sign_up(confirm_username, confirmation_code)
                
                if result['success']:
                    st.success("Email confirmed! Please login")
                else:
                    st.error(result['message'])

        return False
    
    return True
