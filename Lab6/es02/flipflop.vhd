library ieee;
use ieee.std_logic_1164.all;

entity flipflop is
    port(
        clk   : in  std_logic;
        rst_a : in  std_logic; -- async reset
        rst_s : in  std_logic; -- sync reset
        d     : in  std_logic;
        q     : out std_logic
    );
end entity;

architecture beh of flipflop is
begin
    process(clk, rst_a)
    begin
        if rst_a = '1' then
            q <= '0';
        elsif rising_edge(clk) then
            if rst_s = '1' then
                q <= '0';
            else
                q <= d;
            end if;
        end if;
    end process;
end architecture;


